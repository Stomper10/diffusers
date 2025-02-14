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
TODO: Calculate memory consumption for model switching / upload & offload, etc
"""

import os
import time
import math
import shutil
import logging
import argparse
import datasets
import numpy as np
import pandas as pd
import nibabel as nib
import transformers

from packaging import version
from tqdm.auto import tqdm
from monai import transforms
from torchsummary import summary

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
import torchvision

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

import diffusers
from diffusers.utils import check_min_version, is_wandb_available #, deprecate, make_image_grid
from diffusers.optimization import get_scheduler
from diffusers.models.vq_gan_3d import VQGAN, LPIPS, NLayerDiscriminator, NLayerDiscriminator3D
from diffusers.models.vq_gan_3d.vqgan import hinge_d_loss, vanilla_d_loss

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")
logger = get_logger(__name__, log_level="INFO")

@torch.no_grad()
def log_validation(vae, test_dataloader, accelerator, weight_dtype, epoch):
    logger.info("Running validation... ")
    vae_model = accelerator.unwrap_model(vae)

    images = []
    for i, sample in enumerate(test_dataloader):
        if i == 5: # log 10 image pairs (bs=2)
            break
        with torch.autocast(accelerator.device.type, dtype=weight_dtype):
            x = sample["pixel_values"].to(weight_dtype)
            x_recon, _ = vae_model(x)
        images.append(torch.cat([x.cpu(), x_recon.cpu()], axis=0)) #.cpu()

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

def count_activations(model, dummy_input):
    activations = []
    def hook_fn(module, input, output):
        # Add the number of elements in the output to the activations list
        activations.append(output.numel())

    # Register a forward hook for each module to capture the output (activations)
    hooks = []
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d, 
                              nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.Embedding,)):
            hooks.append(layer.register_forward_hook(hook_fn))
    
    # Forward pass through the model
    with torch.no_grad():
        model(dummy_input)
    
    # Remove all the hooks
    for hook in hooks:
        hook.remove()
    
    # Calculate the total number of activations
    total_activations = sum(activations)
    
    return total_activations

def count_parameters(model):
    """
    Ultimate accurate count of the total, trainable, and non-trainable parameters in a model,
    ensuring shared parameters are not double-counted and handling any complex nested submodules.
    
    Args:
        model (torch.nn.Module): The model to count parameters for.
    
    Returns:
        dict: Contains 'total', 'trainable', 'non_trainable' counts of parameters.
    """
    param_count = {
        "total": 0,
        "trainable": 0,
        "non_trainable": 0,
        "by_layer_type": {},  # Added for detailed reporting by layer type
    }

    # Use a set to track shared parameters and avoid double-counting
    seen_params = set()

    for name, param in model.named_parameters():
        param_id = id(param)
        if param_id not in seen_params:
            # Add unique parameter ID to avoid double-counting shared parameters
            seen_params.add(param_id)

            # Count number of elements in the parameter
            num_params = param.numel()

            # Update the total and type-specific counts
            param_count['total'] += num_params
            if param.requires_grad:
                param_count['trainable'] += num_params
            else:
                param_count['non_trainable'] += num_params

            # Track parameters by layer type for detailed reporting
            layer_type = type(param).__name__
            if layer_type not in param_count['by_layer_type']:
                param_count['by_layer_type'][layer_type] = 0
            param_count['by_layer_type'][layer_type] += num_params

            # Optional: print layer-specific details
            #print(f"Layer {name}: {num_params} parameters (Trainable: {param.requires_grad})")

    return param_count


def parse_args():
    parser = argparse.ArgumentParser(description="VQ-GAN training script.")
    parser.add_argument(
        "--axis",
        type=str,
        default=None,
        help=(
            "3D volume's axis that will be processed as the temporal axis."
        ),
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
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=21, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_resolution",
        type=str,
        default="224,160,160",#512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--target_resolution",
        type=str,
        default="224,160,160",#512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--valid_batch_size", type=int, default=16, help="Batch size (per device) for the validation dataloader.",
    )
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
        "--learning_rate_ae",
        type=float,
        default=1e-4, # 1.5e-7, # Reference : Waifu-diffusion-v1-4 config # default=4.5e-8,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_disc",
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
            ' "constant", "constant_with_warmup", "piecewise_constant"]'
        ),
    )
    parser.add_argument(
        "--discriminator_iter_start", type=int, default=10000, help="Number of steps for the warmup before start training discriminator.",
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
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.5, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.9, help="The beta2 parameter for the Adam optimizer.")
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
        "--tracker_project_name",
        type=str,
        default="vae-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args



# Define custom dataset class
class UKB_Dataset(Dataset):
    """
    A PyTorch Dataset for loading UKB images and labels.
    
    Args:
        image_dir (str): Directory containing image files.
        label_dir (str): CSV file containing image IDs and labels.
        transform (callable, optional): Optional transform to apply to samples.
    """
    def __init__(self, image_dir, label_dir, transform=None, axis="c"):
        super().__init__()
        self.data_dir = image_dir
        data_csv = pd.read_csv(label_dir)
        self.image_names = list(data_csv['rel_path'])
        self.transform = transform
        self.axis = axis
        self.image_paths = [os.path.join(self.data_dir, name) for name in self.image_names]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        try:
            image = nib.load(image_path)  # (182, 218, 182)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = image.get_fdata()
        image = torch.from_numpy(image) # Stays on CPU # (182, 218, 182)

        axes_mapping = {
            's': (0, 1, 2),
            'c': (1, 0, 2),
            'a': (2, 1, 0)
        }

        try:
            image = image.permute(*axes_mapping[self.axis]).unsqueeze(0) # (1, 218, 182, 182)
        except KeyError:
            raise ValueError("axis must be one of 'a', 'c', or 's'.")

        # Define target size
        # target_size = (128, 128, 128)  # (D₂, H₂, W₂)
        # # Resize the volume using trilinear interpolation
        # image = F.interpolate(
        #     image,
        #     size=target_size,
        #     mode='trilinear',
        #     align_corners=False
        # )  # Shape: (1, 1, D₂, H₂, W₂)
        # # Remove the batch dimension
        # image = image.squeeze(0).to(torch.float16)  # Shape: (1, D₂, H₂, W₂) # (1, 128, 128, 128)

        sample = {"pixel_values": image.to(torch.float16)}

        if self.transform:
            sample = self.transform(sample) # (1,128,64,64) or (1,224,40,40)
        del image

        return sample



def main():
    args = parse_args()

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
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

    if args.report_to == "wandb" and accelerator.is_main_process: ###
        wandb.init(project=args.tracker_project_name, resume=True) ###

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

    # Load VQ-GAN
    vae = VQGAN(
        embedding_dim=4,
        n_codes=16384,
        n_hiddens=16,
        downsample=(2,2,2),
        image_channels=1,
        restart_thres=1.0,
        no_random_restart=False,
        norm_type="group",
        padding_type="replicate",
        num_groups=32,
    ).to(accelerator.device)

    image_discriminator = NLayerDiscriminator(
        image_channels=1, 
        disc_channels=64, 
        disc_layers=3, 
        norm_layer="BatchNorm2d"
    ).to(accelerator.device)

    video_discriminator = NLayerDiscriminator3D(
        image_channels=1, 
        disc_channels=64, 
        disc_layers=3, 
        norm_layer="BatchNorm3d"
    ).to(accelerator.device)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                logger.info(f"{vae = }") # print model architecture
                for _, model in enumerate(models):
                    if isinstance(model, type(accelerator.unwrap_model(vae))):
                        model.save_pretrained(os.path.join(output_dir, "vqgan"))
                    elif isinstance(model, type(accelerator.unwrap_model(image_discriminator))):
                        model.save_pretrained(os.path.join(output_dir, "image_discriminator"))
                    elif isinstance(model, type(accelerator.unwrap_model(video_discriminator))):
                        model.save_pretrained(os.path.join(output_dir, "video_discriminator"))
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")
                    
                    # make sure to pop weight so that corresponding model is not saved again
                    if weights: ### https://github.com/huggingface/diffusers/issues/2606#issuecomment-1704077101
                        weights.pop()

        def load_model_hook(models, input_dir):
            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                if isinstance(model, type(accelerator.unwrap_model(vae))):
                    load_model = VQGAN.from_pretrained(input_dir, subfolder="vqgan")
                elif isinstance(model, type(accelerator.unwrap_model(image_discriminator))):
                    load_model = NLayerDiscriminator.from_pretrained(input_dir, subfolder="image_discriminator")
                elif isinstance(model, type(accelerator.unwrap_model(video_discriminator))):
                    load_model = NLayerDiscriminator3D.from_pretrained(input_dir, subfolder="video_discriminator")
                else:
                    raise ValueError(f"unexpected load model: {model.__class__}")

                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        vae.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    if args.scale_lr:
        args.learning_rate_ae = (
            args.learning_rate_ae * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        args.learning_rate_disc = (
            args.learning_rate_disc * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes` or `pip install bitsandbytes-windows` for Windows"
            )
        optimizer_cls = bnb.optim.Adam8bit #AdamW8bit
    else:
        optimizer_cls = torch.optim.Adam #AdamW

    optimizer_ae = optimizer_cls(list(vae.encoder.parameters()) +
                                 list(vae.decoder.parameters()) +
                                 list(vae.pre_vq_conv.parameters()) +
                                 list(vae.post_vq_conv.parameters()) +
                                 list(vae.codebook.parameters()),
                                 lr=args.learning_rate_ae, 
                                 betas=(args.adam_beta1, args.adam_beta2))
    optimizer_disc = optimizer_cls(list(image_discriminator.parameters()) +
                                   list(video_discriminator.parameters()),
                                   lr=args.learning_rate_disc,
                                   betas=(args.adam_beta1, args.adam_beta2))

    # Load data
    model_input_size = tuple(int(x) for x in args.model_resolution.split(","))
    target_input_size = tuple(int(x) for x in args.target_resolution.split(","))
    train_transforms = transforms.Compose(
        [
            transforms.SpatialPadd(keys=["pixel_values"], spatial_size=target_input_size, mode="constant"), # SET
            transforms.Resized(keys=["pixel_values"], spatial_size=model_input_size, size_mode="all"),
            transforms.ScaleIntensityd(keys=["pixel_values"], minv=-1.0, maxv=1.0),
            transforms.ThresholdIntensityd(keys=["pixel_values"], threshold=1, above=False, cval=1.0),
            transforms.ThresholdIntensityd(keys=["pixel_values"], threshold=-1, above=True, cval=-1.0),
            transforms.ToTensord(keys=["pixel_values"]),
        ]
    )

    with accelerator.main_process_first():
        if args.data_dir is not None: # args.test_data_dir is not None and args.data_dir is not None:
            train_dataset = UKB_Dataset(args.data_dir, args.train_label_dir, transform=train_transforms, axis=args.axis)
            valid_dataset = UKB_Dataset(args.data_dir, args.valid_label_dir, transform=train_transforms, axis=args.axis)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers, #args.train_batch_size*accelerator.num_processes,
        pin_memory=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        shuffle=False,
        batch_size=args.valid_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler_ae = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_ae,
    )
    lr_scheduler_disc = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_disc,
    )

    # Prepare everything with our `accelerator`.
    vae, image_discriminator, video_discriminator, optimizer_ae, optimizer_disc, lr_scheduler_ae, lr_scheduler_disc, train_dataloader, test_dataloader = accelerator.prepare(
        vae, image_discriminator, video_discriminator, optimizer_ae, optimizer_disc, lr_scheduler_ae, lr_scheduler_disc, train_dataloader, test_dataloader
    )

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

    # ------------------------------ TRAIN ------------------------------ #
    #discriminator_iter_start: int = 50000,
    args.disc_loss_type = "hinge"
    args.image_gan_weight = 1.0
    args.video_gan_weight = 1.0
    args.l1_weight = 4.0
    args.gan_feat_weight = 4.0
    args.perceptual_weight = 4.0

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
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
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

    perceptual_model = LPIPS().to(accelerator.device, dtype=weight_dtype).eval()
    if args.disc_loss_type == 'vanilla':
        disc_loss = vanilla_d_loss
    elif args.disc_loss_type == 'hinge':
        disc_loss = hinge_d_loss

    # # Model memory check
    # dummy_input = torch.ones(1, 1, input_size[0], input_size[1], input_size[2]).float().to(accelerator.device) # 262,144 for 64^3
    # vae.eval()
    # image_discriminator.eval()
    # video_discriminator.eval()

    # vae_activations = count_activations(vae, dummy_input)
    # image_disc_activations = count_activations(image_discriminator, dummy_input[:, :, input_size[0] // 2, :, :])
    # video_disc_activations = count_activations(video_discriminator, dummy_input)

    # print(f"### VQ-GAN's number of activations: {vae_activations:,}")
    # print(f"### image_disc's number of activations: {image_disc_activations:,}")
    # print(f"### video_dics's number of activations: {video_disc_activations:,}")
    
    # Show model architecture and # of params
    param_counts = count_parameters(vae)
    print(f"### VQ-GAN's Total parameters: {param_counts['total']}")
    print(f"### VQ-GAN's Trainable parameters: {param_counts['trainable']}")
    print(f"### VQ-GAN's Non-trainable parameters: {param_counts['non_trainable']}")
    print(f"### VQ-GAN's Parameters by layer type: {param_counts['by_layer_type']}")
    
    param_counts = count_parameters(image_discriminator)
    print(f"### image_disc's Total parameters: {param_counts['total']}")
    print(f"### image_disc's Trainable parameters: {param_counts['trainable']}")
    print(f"### image_disc's Non-trainable parameters: {param_counts['non_trainable']}")
    print(f"### image_disc's Parameters by layer type: {param_counts['by_layer_type']}")

    param_counts = count_parameters(video_discriminator)
    print(f"### video_dics's Total parameters: {param_counts['total']}")
    print(f"### video_dics's Trainable parameters: {param_counts['trainable']}")
    print(f"### video_dics's Non-trainable parameters: {param_counts['non_trainable']}")
    print(f"### video_dics's Parameters by layer type: {param_counts['by_layer_type']}")
    
    # summary(vae, (1, input_size[0], input_size[1], input_size[2]))
    # summary(image_discriminator, (1, input_size[1], input_size[2]))
    # summary(video_discriminator, (1, input_size[0], input_size[1], input_size[2]))

    if model_input_size[0] == 224:
        optimizer_ae = optimizer_cls(list(vae.encoder.parameters()) +
                                     list(vae.decoder.parameters()) +
                                     list(vae.pre_vq_conv.parameters()) +
                                     list(vae.post_vq_conv.parameters()) +
                                     list(vae.codebook.parameters()),
                                     lr=args.learning_rate_ae, 
                                     betas=(args.adam_beta1, args.adam_beta2))
        optimizer_disc = optimizer_cls(list(image_discriminator.parameters()) +
                                       list(video_discriminator.parameters()),
                                       lr=args.learning_rate_disc,
                                       betas=(args.adam_beta1, args.adam_beta2))
        lr_scheduler_ae = get_scheduler(
            "cosine_with_restarts",
            optimizer=optimizer_ae,
            num_warmup_steps=2000,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=1
        )
        lr_scheduler_disc = get_scheduler(
            "cosine_with_restarts",
            optimizer=optimizer_disc,
            num_warmup_steps=2000,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=1
        )
        optimizer_ae, optimizer_disc, lr_scheduler_ae, lr_scheduler_disc = accelerator.prepare(
            optimizer_ae, optimizer_disc, lr_scheduler_ae, lr_scheduler_disc
        )

    # Training
    if accelerator.is_main_process:
        print("### Start training ###")
    for epoch in range(first_epoch, args.num_train_epochs):
        #train_loss = 0.0
        logger.info(f"{epoch = }")

        for step, batch in enumerate(train_dataloader):
            
            start_time = time.time()

            if step % 2 == 0:
                vae.train()
                image_discriminator.eval()
                video_discriminator.eval()

                # Accumulate gradients
                with accelerator.accumulate(vae):
                    with torch.autocast(accelerator.device.type, dtype=weight_dtype):
                        x = batch["pixel_values"].to(weight_dtype)
                        B, C, T, H, W = x.shape

                        x_recon, vq_output = vae(x)

                        # Compute reconstruction loss
                        recon_loss = F.l1_loss(x_recon, x) * args.l1_weight
                        # Compute commitment loss
                        commitment_loss = vq_output['commitment_loss']

                        # Selects one random 2D image from each 3D Image
                        frame_idx = torch.randint(0, T, [B]).to(accelerator.device) #.cuda()
                        frame_idx_selected = frame_idx.reshape(-1, 1, 1, 1, 1).repeat(1, C, 1, H, W)
                        frames = torch.gather(x, 2, frame_idx_selected).squeeze(2)
                        frames_recon = torch.gather(x_recon, 2, frame_idx_selected).squeeze(2)
                        del frame_idx, frame_idx_selected

                        with torch.no_grad():
                            perceptual_loss = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)
                            if args.perceptual_weight > 0:
                                perceptual_loss = perceptual_model(frames, frames_recon).mean() * args.perceptual_weight
                        
                        # Discriminator loss (turned on after a certain epoch)
                        ae_loss = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)
                        gan_feat_loss = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)
                        g_image_loss = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)
                        g_video_loss = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)
                        image_gan_feat_loss = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)
                        video_gan_feat_loss = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)

                        if global_step > args.discriminator_iter_start:
                            with torch.no_grad():
                                logits_image_fake, pred_image_fake = image_discriminator(frames_recon)
                                logits_video_fake, pred_video_fake = video_discriminator(x_recon)
                                g_image_loss = -torch.mean(logits_image_fake)
                                g_video_loss = -torch.mean(logits_video_fake)
                                ae_loss = args.image_gan_weight * g_image_loss + args.video_gan_weight * g_video_loss

                                # GAN feature matching loss - tune features such that we get the same prediction result on the discriminator
                                image_gan_feat_loss, video_gan_feat_loss = 0, 0
                                feat_weights = 1 #4.0 / (3 + 1)
                                if args.image_gan_weight > 0:
                                    logits_image_real, pred_image_real = image_discriminator(frames)
                                    for i in range(len(pred_image_fake)-1):
                                        image_gan_feat_loss += feat_weights * \
                                            F.l1_loss(pred_image_fake[i], pred_image_real[i].detach(
                                            ))
                                if args.video_gan_weight > 0:
                                    logits_video_real, pred_video_real = video_discriminator(x)
                                    for i in range(len(pred_video_fake)-1):
                                        video_gan_feat_loss += feat_weights * \
                                            F.l1_loss(pred_video_fake[i], pred_video_real[i].detach(
                                            ))
                                gan_feat_loss = args.gan_feat_weight * (image_gan_feat_loss + video_gan_feat_loss)
                            
                            del logits_image_fake, pred_image_fake, logits_video_fake, pred_video_fake
                            del logits_image_real, pred_image_real, logits_video_real, pred_video_real
                        
                        del x_recon, frames, frames_recon

                        loss = recon_loss + commitment_loss + ae_loss + perceptual_loss + gan_feat_loss

                        if not torch.isfinite(loss):
                            logger.info("\nWARNING: non-finite loss.")
                            continue
                            
                        optimizer_ae.zero_grad(set_to_none=True)
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(vae.parameters(), args.max_grad_norm)
                        optimizer_ae.step()
                        lr_scheduler_ae.step()

                #if step % args.gradient_accumulation_steps == 0:
                logs = {
                    "ae_step_loss": loss.detach().item(),
                    "lr": lr_scheduler_ae.get_last_lr()[0],
                    "perceptual_loss": perceptual_loss.detach().item(),
                    "recon_loss": recon_loss.detach().item(),
                    "commitment_loss": vq_output['commitment_loss'].detach().item(),
                    "perplexity": vq_output['perplexity'].detach().item(),
                    "g_image_loss": g_image_loss.detach().item(),
                    "g_video_loss": g_video_loss.detach().item(),
                    "ae_loss": ae_loss.detach().item(),
                    "image_gan_feat_loss": image_gan_feat_loss.detach().item(),
                    "video_gan_feat_loss": video_gan_feat_loss.detach().item(),
                    "gan_feat_loss": gan_feat_loss.detach().item(),
                }
                accelerator.log(logs)

            else: # Update Discriminators on odd steps
                loss = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)
                d_image_loss = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)
                d_video_loss = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)

                if global_step > args.discriminator_iter_start:
                    vae.eval()
                    image_discriminator.train()
                    video_discriminator.train()

                    # Accumulate gradients
                    with accelerator.accumulate(image_discriminator), accelerator.accumulate(video_discriminator):
                        with torch.autocast(accelerator.device.type, dtype=weight_dtype):
                            x = batch["pixel_values"].to(weight_dtype)
                            B, C, T, H, W = x.shape
                            with torch.no_grad():
                                x_recon, vq_output = vae(x)
                            del vq_output

                            # Selects one random 2D image from each 3D Image
                            frame_idx = torch.randint(0, T, [B]).to(accelerator.device) #.cuda()
                            frame_idx_selected = frame_idx.reshape(-1, 1, 1, 1, 1).repeat(1, C, 1, H, W)
                            frames = torch.gather(x, 2, frame_idx_selected).squeeze(2)
                            frames_recon = torch.gather(x_recon, 2, frame_idx_selected).squeeze(2)

                            # Train discriminator
                            logits_image_real, pred_image_real = image_discriminator(frames.detach())
                            logits_video_real, pred_video_real = video_discriminator(x.detach())

                            logits_image_fake, pred_image_fake = image_discriminator(frames_recon.detach())
                            logits_video_fake, pred_video_fake = video_discriminator(x_recon.detach())

                            d_image_loss = disc_loss(logits_image_real, logits_image_fake)
                            d_video_loss = disc_loss(logits_video_real, logits_video_fake)
                            discloss = args.image_gan_weight * d_image_loss + args.video_gan_weight * d_video_loss
                            
                            del x_recon, frames, frames_recon
                            del logits_image_fake, pred_image_fake, logits_video_fake, pred_video_fake
                            del logits_image_real, pred_image_real, logits_video_real, pred_video_real

                            loss = discloss

                            if not torch.isfinite(loss):
                                logger.info("\nWARNING: non-finite loss.")
                                continue

                            optimizer_disc.zero_grad(set_to_none=True)
                            accelerator.backward(loss)
                            if accelerator.sync_gradients:
                                disc_params = list(image_discriminator.parameters()) + list(video_discriminator.parameters())
                                accelerator.clip_grad_norm_(disc_params, args.max_grad_norm)
                            optimizer_disc.step()
                            lr_scheduler_disc.step()
                            
                #if step % args.gradient_accumulation_steps == 0:
                logs = {    
                    "disc_step_loss": loss.detach().item(),
                    "lr": lr_scheduler_disc.get_last_lr()[0],
                    "d_image_loss": d_image_loss.detach().item(),
                    "d_video_loss": d_video_loss.detach().item(),
                }
                accelerator.log(logs)

            end_time = time.time() ###
            if accelerator.is_main_process:
                print(f"### Training step elasped: {end_time - start_time}", flush=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

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
                                    try:
                                        shutil.rmtree(removing_checkpoint)
                                    except OSError as e:
                                        logger.error(f"Error removing checkpoint {removing_checkpoint}: {e}")

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                    log_validation(vae, test_dataloader, accelerator, weight_dtype, epoch)

                    # validation
                    if accelerator.is_main_process:
                        print("### Start validation ###")
                    with torch.no_grad():
                        vae.eval()
                        image_discriminator.eval()
                        video_discriminator.eval()
                        valid_loss = 0.0
                        for step, batch in enumerate(test_dataloader):
                            with torch.autocast(accelerator.device.type, dtype=weight_dtype):
                                x = batch["pixel_values"].to(weight_dtype)
                                B, C, T, H, W = x.shape
                                x_recon, val_vq_output = vae(x)
                                #del val_vq_output

                                val_recon_loss = F.l1_loss(x_recon, x) * args.l1_weight
                                # Selects one random 2D image from each 3D Image
                                frame_idx = torch.randint(0, T, [B]).to(accelerator.device) #.cuda()
                                frame_idx_selected = frame_idx.reshape(-1, 1, 1, 1, 1).repeat(1, C, 1, H, W)
                                frames = torch.gather(x, 2, frame_idx_selected).squeeze(2)
                                frames_recon = torch.gather(x_recon, 2, frame_idx_selected).squeeze(2)

                                val_perceptual_loss = perceptual_model(frames, frames_recon).mean() * args.perceptual_weight
                                
                                del x_recon, frames, frames_recon

                                # Gather the losses across all processes for logging (if we use distributed training).
                                val_loss = val_recon_loss + val_perceptual_loss

                                if not torch.isfinite(val_loss):
                                    logger.info("\nWARNING: non-finite loss.")
                                    continue

                                # Gather the losses across all processes for logging (if we use distributed training).
                                gathered_valid_loss = accelerator.gather(val_loss)
                                valid_loss += gathered_valid_loss.sum()

                        logs = {
                            "valid_loss": valid_loss.detach().item() / len(test_dataloader.dataset),
                            "valid_recon_loss": val_recon_loss.detach().item(),
                            "valid_perceptual_loss": val_perceptual_loss.detach().item(),
                            "valid_perplexity": val_vq_output['perplexity'].detach().item(),
                            "valid_commitment_loss": val_vq_output['commitment_loss'].detach().item(),
                        }
                        accelerator.log(logs)

            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":
    main()