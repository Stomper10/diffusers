#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
TODO: Implement ZeRO3
"""

import argparse
import logging
import math
import os
import shutil
from pathlib import Path

import accelerate
import datasets
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
#from torchvision import transforms
from tqdm.auto import tqdm
from transformers.utils import ContextManagers
from torch.utils.data import Dataset

import diffusers
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

import torchvision
import lpips
from PIL import Image

from acsconv.converters import ACSConverter
from monai import transforms
import time

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}

torch.backends.cudnn.benchmark = True ###
nb_acts, nb_inputs, nb_params = 0, 0, 0

def save_model_card(
    args,
    repo_id: str,
    images: list = None,
    repo_folder: str = None,
):
    img_str = ""
    if len(images) > 0:
        image_grid = make_image_grid(images, 1, len(images))
        image_grid.save(os.path.join(repo_folder, "val_imgs_grid.png"))
        img_str += "![val_imgs_grid](./val_imgs_grid.png)\n"

    model_description = f"""
# VAE finetuning - {repo_id}

This pipeline was finetuned from **{args.pretrained_model_name_or_path}** on the **{args.dataset_name}** dataset. Below are some example images generated with the finetuned VAE: \n
{img_str}

## Training info

These are the key hyperparameters used during training:

* Epochs: {args.num_train_epochs}
* Learning rate: {args.learning_rate}
* Batch size: {args.train_batch_size}
* Gradient accumulation steps: {args.gradient_accumulation_steps}
* Image resolution: {args.resolution}
* Mixed-precision: {args.mixed_precision}

"""
    wandb_info = ""
    if is_wandb_available():
        wandb_run_url = None
        if wandb.run is not None:
            wandb_run_url = wandb.run.url

    if wandb_run_url is not None:
        wandb_info = f"""
More information on all the CLI arguments and the environment are available on your [`wandb` run page]({wandb_run_url}).
"""

    model_description += wandb_info

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=args.pretrained_model_name_or_path,
        model_description=model_description,
        inference=True,
    )

    tags = ["stable-diffusion", "stable-diffusion-diffusers", "text-to-image", "diffusers", "diffusers-training", "vae"]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


@torch.no_grad()
def log_validation(vae, test_dataloader, accelerator, weight_dtype, epoch):
    logger.info("Running validation... ")

    vae_model = accelerator.unwrap_model(vae)
    
    images = []
    for i, sample in enumerate(test_dataloader):
        if i < 30: # log 10 images
            autocast_ctx = torch.autocast(accelerator.device.type)
            with autocast_ctx:
                x = sample["pixel_values"].to(weight_dtype)
                reconstructions = vae_model(x).sample

                x = rgb_to_grayscale_3d(x)
                reconstructions = rgb_to_grayscale_3d(reconstructions)
                
            images.append(
                torch.cat([x.cpu(), reconstructions.cpu()], axis=0) #.cpu()
            )

    print("### x.shape:", x.shape, flush=True)
    print("### reconstructions.shape:", reconstructions.shape, flush=True)

    for tracker in accelerator.trackers:
        tracker.name = "wandb"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("Original (left), Reconstruction (right)", np_images, epoch)
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "Original (left), Reconstruction (right) - 1": [
                        wandb.Image(torchvision.utils.make_grid(image[:, :, image.shape[2] // 2, :, :]).permute(1, 2, 0))
                        for _, image in enumerate(images)
                    ],
                    "Original (left), Reconstruction (right) - 2": [
                        wandb.Image(torchvision.utils.make_grid(image[:, :, :, image.shape[3] // 2, :]).permute(1, 2, 0))
                        for _, image in enumerate(images)
                    ],
                    "Original (left), Reconstruction (right) - 3": [
                        wandb.Image(torchvision.utils.make_grid(image[:, :, :, :, image.shape[4] // 2]).permute(1, 2, 0))
                        for _, image in enumerate(images)
                    ]
                }
            )
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    del vae_model
    del images
    torch.cuda.empty_cache()

    #return images


def rgb_to_grayscale_3d(voxel_image):
    """
    Convert a 3D voxel image from RGB to grayscale.
    
    Args:
    - voxel_image (torch.Tensor): The input tensor with shape (batch_size, 3, depth, height, width)
    
    Returns:
    - torch.Tensor: The grayscale voxel image with shape (batch_size, 1, depth, height, width)
    """
    if voxel_image.shape[1] != 3:
        raise ValueError("Input voxel image must have 3 channels (RGB).")
    
    # Define the RGB to grayscale conversion weights
    weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float16, device=voxel_image.device)
    
    # Reshape weights for broadcasting
    weights = weights.view(1, 3, 1, 1, 1)
    
    # Apply the weights to the RGB channels
    grayscale_image = (voxel_image * weights).sum(dim=1, keepdim=True)
    
    return grayscale_image


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a VAE training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--train_label_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training label. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--valid_label_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the validation label. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    # parser.add_argument(
    #     "--image_column", type=str, default="image", help="The column of the dataset containing an image.",
    # )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=21, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=str,
        default="160,224,160",#512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--valid_batch_size", type=int, default=16, help="Batch size (per device) for the validation dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4, # 1.5e-7, # Reference : Waifu-diffusion-v1-4 config # default=4.5e-8,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=True,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="vae-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=20,
        help="Number of images to remove from training set to be used as validation.",
    )
    parser.add_argument(
        "--kl_scale",
        type=float,
        default=1e-6,
        help="Scaling factor for the Kullback-Leibler divergence penalty term.",
    )
    parser.add_argument(
        "--lpips_scale",
        type=float,
        default=5e-1,
        help="Scaling factor for the LPIPS metric",
    )
    parser.add_argument(
        "--lpips_start",
        type=int,
        default=50001,
        help="Start for the LPIPS metric",
    )
    parser.add_argument(
        "--tile_sample_size",
        type=int,
        default=None,
        help="Start for the LPIPS metric",
    )
    parser.add_argument(
        "--slicing",
        action="store_true",
        help="Enable sliced VAE (process single batch at a time)",
    )
    parser.add_argument(
        "--tiling",
        action="store_true",
        help="Enable tiling VAE (process divided image)",
    )

    
    args = parser.parse_args()

    # args.mixed_precision='fp16'
    # args.pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"
    # args.dataset_name="yeonsikc/sample_repeat"
    # args.seed=21
    # args.train_batch_size=1
    # args.num_train_epochs=100
    # args.learning_rate=1e-07
    # args.output_dir="/app/output_vae"
    # args.report_to='wandb'
    # args.push_to_hub=True
    # args.validation_epochs=1
    # args.resolution=128
    # args.use_8bit_adam=False

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")
    
    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args

# train_transforms = transforms.Compose(
#     [
#         transforms.Resize(
#             (128,128), interpolation=transforms.InterpolationMode.BILINEAR
#         ),
#         # transforms.RandomCrop(128),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5]),
#     ]
# )

# def preprocess(examples):
#     images = [image.convert("RGB") for image in examples["image"]]
#     examples["pixel_values"] = [train_transforms(image) for image in images]
#     return examples

# def collate_fn(examples):
#     pixel_values = torch.stack([example["pixel_values"] for example in examples])
#     pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
#     return {"pixel_values": pixel_values}

def main():
    args = parse_args()

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load vae
    try:
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.non_ema_revision, variant=args.variant
        )
        if args.tiling and args.tile_sample_size:
            vae.tile_sample_min_size = args.tile_sample_size
            vae.tile_latent_min_size = int(args.tile_sample_size / 8) ### hard-coded assuming config sample size is 512
        vae = ACSConverter(vae).to(accelerator.device) ###
        #vae.to(memory_format=torch.channels_last) ###
        print("### ACS applied ###", flush=True)
    except:
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, revision=args.non_ema_revision, variant=args.variant
        )

    #vae.requires_grad_(True)
    vae.train()
    
    if args.use_ema:
        try:
            ema_vae = AutoencoderKL.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
            ema_vae = ACSConverter(ema_vae) ###
            print("### ACS applied ema ###", flush=True)
        except:
            ema_vae = AutoencoderKL.from_pretrained(
                args.pretrained_model_name_or_path, revision=args.revision, variant=args.variant)
        ema_vae = EMAModel(ema_vae.parameters(), model_cls=AutoencoderKL, model_config=ema_vae.config)
        
    #vae_params = vae.parameters()

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_vae.save_pretrained(os.path.join(output_dir, "vae_ema"))

                logger.info(f"{vae = }")
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "vae"))

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights: ### https://github.com/huggingface/diffusers/issues/2606#issuecomment-1704077101
                        weights.pop()

                # vae = vae[0]
                # vae.save_pretrained(os.path.join(output_dir, "vae"))
                # weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "vae_ema"), AutoencoderKL)
                ema_vae.load_state_dict(load_model.state_dict())
                ema_vae.to(accelerator.device)
                del load_model

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = AutoencoderKL.from_pretrained(input_dir, subfolder="vae")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

            # load diffusers style into model
            # load_model = AutoencoderKL.from_pretrained(input_dir, subfolder="vae")
            # vae = vae[0]
            # vae.register_to_config(**load_model.config)
            # vae.load_state_dict(load_model.state_dict())
            # del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        vae.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True ###

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        
    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes` or `pip install bitsandbytes-windows` for Windows"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        vae.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # if args.dataset_name is not None:
    #     # Downloading and loading a dataset from the hub.
    #     dataset = load_dataset(
    #         args.dataset_name,
    #         args.dataset_config_name,
    #         cache_dir=args.cache_dir,
    #         data_dir=args.train_data_dir,
    #     )
    # else:
    #     data_files = {}
    #     if args.train_data_dir is not None:
    #         data_files["train"] = os.path.join(args.train_data_dir, "**")
    #     dataset = load_dataset(
    #         "imagefolder",
    #         data_files=data_files,
    #         cache_dir=args.cache_dir,
    #     )

    #     test_dir = os.path.join(args.test_data_dir, "**")
    #     test_dataset = load_dataset(
    #         "imagefolder",
    #         data_files=test_dir,
    #         cache_dir=args.cache_dir,
    #     )

    class UKB_Dataset(Dataset):
        def __init__(self, image_dir, label_dir, transform=None, device=None):
            super(UKB_Dataset, self).__init__()
            self.data_dir = image_dir
            data_csv = pd.read_csv(label_dir)
            self.image_names = list(data_csv['id'])
            # self.labels = list(data_csv['age'])
            # print("images_names: ", len(self.image_names), self.image_names[-1])
            # print("labels: ", len(self.labels), self.labels[-1])
            self.transform = transform
            self.device = device
            del data_csv

        def __len__(self):
            return len(self.image_names)

        def __getitem__(self, index):
            image_name = self.image_names[index]
            #label = self.labels[index]
            # Load the image
            image = np.load(os.path.join(self.data_dir, 'final_array_128_full_' + str(image_name) + '.npy')).astype(np.float16) # (128,128,128,1)
            #print("image shape:", image.shape)
            image = torch.from_numpy(image).type(torch.float16) ###
            image = image.permute(3, 0, 1, 2) # (1,128,128,128)
            #image = np.load(os.path.join(self.data_dir, 'final_array_128_full_' + str(image_name) + '.npy')).astype(np.uint8)[:,64,:,0] # (128,128,128,1) -> (128,128)
            #image = Image.fromarray(image, "L")
            #image = image.repeat(3, 1, 1, 1) # (3,128,128,128)

            sample = dict()
            sample["pixel_values"] = image
            del image

            if self.transform:
                sample = self.transform(sample) # (128,128) -> (1,160,160)
                # sample["pixel_values"] = sample["pixel_values"][0,:,:,:] # pseudo color channel (64,64,64)
            sample["pixel_values"] = sample["pixel_values"].repeat(3, 1, 1, 1)
            #image = torch.from_numpy(image).to(memory_format=torch.contiguous_format).float()
            #image = image.repeat(3, 1, 1, 1) # (1,160,160) -> (3,160,160)
            # age = torch.tensor(label, dtype=torch.float32)

            #sample = dict()
            #sample["pixel_values"] = image
            # sample["label"] = age
            #del image
            
            #print("sample shape:", sample["pixel_values"].shape)
            #print("sample dtype:", sample["pixel_values"].dtype)
            return sample
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # # Preprocessing the datasets.
    # # We need to tokenize inputs and targets.
    # column_names = dataset["train"].column_names

    # # 6. Get the column names for input/target.
    # dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    # if args.image_column is None:
    #     image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    # else:
    #     image_column = args.image_column
    #     if image_column not in column_names:
    #         raise ValueError(
    #             f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
    #         )
    input_size = tuple(int(x) for x in args.resolution.split(","))
    # train_transforms = transforms.Compose(
    #     [
    #         #transforms.Resize((input_size[0], input_size[1], input_size[2]), interpolation=transforms.InterpolationMode.BILINEAR),
    #         #transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
    #         #transforms.RandomCrop(args.resolution),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.5], [0.5]),
    #     ]
    # )

    train_transforms = transforms.Compose(
        [
            #transforms.LoadImaged(keys=["pixel_values"]),
            #transforms.EnsureChannelFirstd(keys=["pixel_values"]),
            transforms.ScaleIntensityd(keys=["pixel_values"], minv=-1.0, maxv=1.0),
            transforms.Resized(keys=["pixel_values"], spatial_size=input_size, size_mode="all"),
            transforms.ThresholdIntensityd(keys=["pixel_values"], threshold=1, above=False, cval=1.0),
            transforms.ThresholdIntensityd(keys=["pixel_values"], threshold=-1, above=True, cval=-1.0),
            transforms.ToTensord(keys=["pixel_values"]),
            # transforms.Lambdad(keys="pixel_values", func=lambda x: x[channel, :, :, :]),
            # transforms.AddChanneld(keys=["pixel_values"]),
            # transforms.EnsureTyped(keys=["pixel_values"]),
            # transforms.Orientationd(keys=["pixel_values"], axcodes="RAS"),
            # transforms.Spacingd(keys=["pixel_values"], pixdim=(2.4, 2.4, 2.2), mode=("bilinear")),
            # transforms.CenterSpatialCropd(keys=["pixel_values"], roi_size=(64, 64, 64)),
            # transforms.ScaleIntensityRangePercentilesd(keys=["pixel_values"], lower=0, upper=99.5, b_min=0, b_max=1),
            # transforms.NormalizeIntensityd(keys=["pixel_values"])
        ]
    )

    # def preprocess(examples):
    #     images = [image.convert("RGB") for image in examples[image_column]]
    #     examples["pixel_values"] = [train_transforms(image) for image in images]
    #     return examples

    with accelerator.main_process_first():
        # Load test data from test_data_dir
        if args.data_dir is not None: # args.test_data_dir is not None and args.data_dir is not None:
            # logger.info(f"load test data from {args.test_data_dir}")
            # test_dir = os.path.join(args.test_data_dir, "**")
            # test_dataset = load_dataset(
            #     "imagefolder",
            #     data_files=test_dir,
            #     cache_dir=args.cache_dir,
            # )
            # # Set the training transforms
            # train_dataset = dataset["train"].with_transform(preprocess)
            # test_dataset = test_dataset["test"].with_transform(preprocess)
            train_dataset = UKB_Dataset(args.data_dir, args.train_label_dir, transform=train_transforms, device=accelerator.device)
            valid_dataset = UKB_Dataset(args.data_dir, args.valid_label_dir, transform=train_transforms, device=accelerator.device)
        # Load train/test data from train_data_dir
        # elif "test" in dataset.keys():
        #     train_dataset = dataset["train"].with_transform(preprocess)
        #     test_dataset = dataset["test"].with_transform(preprocess)
        # # Split into train/test
        # else:
        #     dataset = dataset["train"].train_test_split(test_size=args.test_samples)        
        #     # Set the training transforms
        #     train_dataset = dataset["train"].with_transform(preprocess)
        #     test_dataset = dataset["test"].with_transform(preprocess)

    # def collate_fn(examples):
    #     pixel_values = torch.stack([example["pixel_values"] for example in examples])
    #     pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    #     return {"pixel_values": pixel_values}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        #collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers, #args.train_batch_size*accelerator.num_processes,
        pin_memory=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        shuffle=False,
        #collate_fn=collate_fn,
        batch_size=args.valid_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True
    )

    # test_dataloader = torch.utils.data.DataLoader(
    #     test_dataset, shuffle=False, collate_fn=collate_fn, batch_size=args.train_batch_size, num_workers=1,#args.train_batch_size*accelerator.num_processes,
    # )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    vae, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        vae, optimizer, train_dataloader, test_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_vae.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32 #16
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)
        
    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # ------------------------------ TRAIN ------------------------------ #
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num test samples = {len(valid_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            # dirs = os.listdir(args.resume_from_checkpoint)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            #accelerator.load_state(os.path.join(path)) #kiml
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            # resume_global_step = global_step * args.gradient_accumulation_steps
            # first_epoch = global_step // num_update_steps_per_epoch
            # resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    lpips_loss_fn = lpips.LPIPS(net="alex").to(accelerator.device, dtype=weight_dtype)
    lpips_loss_fn.requires_grad_(False)

    #accelerator.init_deepspeed() ###


    #################### Model check #####################


    # autoencoder memory check
    def count_output_act(m, input, output):
        global nb_acts
        nb_acts += output.nelement()

    for module in vae.modules():
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear) or isinstance(module, nn.GroupNorm):
            module.register_forward_hook(count_output_act)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    nb_params = sum([p.nelement() for p in vae.parameters()])

    vae_input = torch.ones(1, 3, 64, 64, 64).float().to(accelerator.device) # 716,800
    vae.eval() # 13,766,248
    encoded = vae(vae_input)
    nb_inputs = vae_input.nelement()

    print('input elem: {}, param elem: {}, forward act: {}, mem usage: {}GB'.format(
        nb_inputs, nb_params, nb_acts, (nb_inputs+nb_params+nb_acts)*4/1024**3))
    print("{:.2f} GB".format(torch.cuda.memory_allocated() / 1024 ** 3))
    print("Autoencdoer parameters:", count_parameters(vae))
    summary(vae, (3, 64, 64, 64))
    # summary(vae, 3, 64, 64, 64)


    for epoch in range(first_epoch, args.num_train_epochs):
        vae.train()
        if args.slicing:
            vae.enable_slicing()
        if args.tiling:
            vae.enable_tiling()
        train_loss = 0.0
        logger.info(f"{epoch = }")

        for step, batch in enumerate(train_dataloader):
            start_time = time.time() ###
            # Skip steps until we reach the resumed step
            # if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
            #     if step % args.gradient_accumulation_steps == 0:
            #         progress_bar.update(1)
            #     continue
            with accelerator.accumulate(vae):
                target = batch["pixel_values"].to(weight_dtype) #.to(memory_format=torch.channels_last_3d) #.to(weight_dtype) ###

                # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoder_kl.py
                if accelerator.num_processes > 1:
                    posterior = vae.module.encode(target).latent_dist
                else:
                    posterior = vae.encode(target).latent_dist # code run this line
                
                # z = mean                      if posterior.mode()
                # z = mean + variable*epsilon   if posterior.sample()
                z = posterior.sample() # Not mode()
                if accelerator.num_processes > 1:
                    pred = vae.module.decode(z).sample
                else:
                    pred = vae.decode(z).sample

                if global_step > args.lr_warmup_steps:
                    kl_loss = posterior.kl().mean()
                    print("### kl loss", kl_loss, flush=True)
                
                # if global_step > args.mse_start:
                #     pixel_loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
                # else:
                #     pixel_loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
                
                # print("### pred shape:", pred.shape, flush=True)
                # print("### target shape:", target.shape, flush=True)

                pred = rgb_to_grayscale_3d(pred)
                target = rgb_to_grayscale_3d(target)
                print("### pred.shape:", pred.shape, flush=True)
                print("### target.shape:", target.shape, flush=True)
                mse_loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
                
                with torch.no_grad():
                    # for i, (pre, tar) in enumerate(zip(pred, target)):
                    lpips_loss = 0
                    # print("### pred[:, :, :, :, i] shape:", pred[..., 0].shape)
                    # print("### pred[:, :, :, i, :] shape:", pred[..., 0, :].shape)
                    # print("### pred[:, :, i, :, :] shape:", pred[..., 0, :, :].shape)
                    for i in range(pred.shape[-1]):
                        lpips_loss += lpips_loss_fn(pred[:, :, :, :, i].to(dtype=weight_dtype), 
                                                    target[:, :, :, :, i].to(dtype=weight_dtype)).sum()#.mean()
                    for i in range(pred.shape[-2]):
                        lpips_loss += lpips_loss_fn(pred[:, :, :, i, :].to(dtype=weight_dtype), 
                                                    target[:, :, :, i, :].to(dtype=weight_dtype)).sum()#.mean()
                    for i in range(pred.shape[-3]):
                        lpips_loss += lpips_loss_fn(pred[:, :, i, :, :].to(dtype=weight_dtype), 
                                                    target[:, :, i, :, :].to(dtype=weight_dtype)).sum()#.mean()
                    lpips_loss /= ((pred.shape[-1] + pred.shape[-2] + pred.shape[-3])*pred.shape[0])
                    
                    #lpips_loss = lpips_loss_fn(pred.to(dtype=weight_dtype), target).mean()
                    if not torch.isfinite(lpips_loss):
                        lpips_loss = torch.tensor(0)

                if global_step > args.lr_warmup_steps:
                    loss = (mse_loss + args.lpips_scale * lpips_loss + args.kl_scale * kl_loss)
                else:
                    loss = (mse_loss + args.lpips_scale * lpips_loss) # + args.kl_scale * kl_loss)

                if not torch.isfinite(loss):
                    # pred_mean = pred.mean()
                    # target_mean = target.mean()
                    logger.info("\nWARNING: non-finite loss, ending training ")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                    
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(vae.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                # for param in vae.parameters():
                #     param.grad = None

            end_time = time.time() ###
            print(f"### Training step elasped: {end_time - start_time}", flush=True) ###
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_vae.step(vae.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                    log_validation(vae, test_dataloader, accelerator, weight_dtype, epoch)

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "mse_loss": mse_loss.detach().item(),
                "lpips_loss": lpips_loss.detach().item(),
                #"kl_loss": kl_loss.detach().item(),
            }
            accelerator.log(logs)
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    #if accelerator.is_main_process:
    #if global_step % args.validation_epochs == 0: ### global_step
    if args.use_ema:
        # Store the VAE parameters temporarily and load the EMA parameters to perform inference.
        ema_vae.store(vae.parameters())
        ema_vae.copy_to(vae.parameters())

    if args.slicing:
        vae.disable_slicing()
    if args.tiling:
        vae.disable_tiling()

    with torch.no_grad():
        vae.eval()
        valid_loss = 0.0
        for step, batch in enumerate(test_dataloader):
            with accelerator.accumulate(vae):
                target = batch["pixel_values"].to(weight_dtype) #.to(memory_format=torch.channels_last_3d) #.to(weight_dtype) ###

                # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoder_kl.py
                if accelerator.num_processes > 1:
                    posterior = vae.module.encode(target).latent_dist
                else:
                    posterior = vae.encode(target).latent_dist # code run this line
                
                # z = mean                      if posterior.mode()
                # z = mean + variable*epsilon   if posterior.sample()
                z = posterior.sample() # Not mode()
                if accelerator.num_processes > 1:
                    pred = vae.module.decode(z).sample
                else:
                    pred = vae.decode(z).sample

                kl_loss = posterior.kl().mean()
                
                # if global_step > args.mse_start:
                #     pixel_loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
                # else:
                #     pixel_loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
                
                # print("### pred shape:", pred.shape, flush=True)
                # print("### target shape:", target.shape, flush=True)

                pred = rgb_to_grayscale_3d(pred)
                target = rgb_to_grayscale_3d(target)
                
                mse_loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
                
                #with torch.no_grad():
                # for i, (pre, tar) in enumerate(zip(pred, target)):
                lpips_loss = 0
                # print("### pred[..., i].squeeze(dim=-1) shape:", pred[..., 0].squeeze(dim=-1).shape)
                # print("### pred[..., i, :].squeeze(dim=-2) shape:", pred[..., 0, :].squeeze(dim=-2).shape)
                # print("### pred[..., i, :, :].squeeze(dim=-3) shape:", pred[..., 0, :, :].squeeze(dim=-3).shape)
                for i in range(pred.shape[-1]):
                    lpips_loss += lpips_loss_fn(pred[:, :, :, :, i].to(dtype=weight_dtype), 
                                                target[:, :, :, :, i].to(dtype=weight_dtype)).sum()#.mean()
                for i in range(pred.shape[-2]):
                    lpips_loss += lpips_loss_fn(pred[:, :, :, i, :].to(dtype=weight_dtype), 
                                                target[:, :, :, i, :].to(dtype=weight_dtype)).sum()#.mean()
                for i in range(pred.shape[-3]):
                    lpips_loss += lpips_loss_fn(pred[:, :, i, :, :].to(dtype=weight_dtype), 
                                                target[:, :, i, :, :].to(dtype=weight_dtype)).sum()#.mean()
                lpips_loss /= ((pred.shape[-1] + pred.shape[-2] + pred.shape[-3])*pred.shape[0])
                
                #lpips_loss = lpips_loss_fn(pred.to(dtype=weight_dtype), target).mean()
                if not torch.isfinite(lpips_loss):
                    lpips_loss = torch.tensor(0)

                loss = (
                    mse_loss + args.lpips_scale * lpips_loss + args.kl_scale * kl_loss
                )

                if not torch.isfinite(loss):
                    # pred_mean = pred.mean()
                    # target_mean = target.mean()
                    logger.info("\nWARNING: non-finite loss, ending validation ")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.valid_batch_size)).mean()
                valid_loss += avg_loss.item() # / args.gradient_accumulation_steps

            if accelerator.sync_gradients:
                accelerator.log({"valid_loss": valid_loss}, step=global_step)
                valid_loss = 0.0

        logs = {
            "valid_step_loss": loss.detach().item(),
            "valid_mse_loss": mse_loss.detach().item(),
            "valid_lpips_loss": lpips_loss.detach().item(),
            "valid_kl_loss": kl_loss.detach().item(),
        }
        accelerator.log(logs)

    #with torch.no_grad():
    #log_validation(args, repo_id, test_dataloader, vae, accelerator, weight_dtype, epoch)
    log_validation(vae, test_dataloader, accelerator, weight_dtype, epoch)
    if args.use_ema:
        # Switch back to the original parameters.
        ema_vae.restore(vae.parameters())

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        vae = unwrap_model(vae) # accelerator.unwrap_model(vae)
        if args.use_ema:
            ema_vae.copy_to(vae.parameters())
        vae.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    main()
    