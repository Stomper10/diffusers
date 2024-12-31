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
TODO: Diffusers scheduler? SD3? (flow matching)
TODO: increase cond_dim if add conditions
"""

import os
import time
import math
import shutil
import random
import logging
import argparse
import datasets
import numpy as np
import pandas as pd
import transformers
#import lpips

from packaging import version
from tqdm.auto import tqdm
from monai import transforms
from torchsummary import summary
from einops_exts import check_shape #, rearrange_many
#from PIL import Image
#from pathlib import Path
#from omegaconf import OmegaConf
#from datasets import load_dataset
#from transformers.utils import ContextManagers
#from huggingface_hub import create_repo , upload_folder

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
import torchvision
#from torch.optim.lr_scheduler import _LRScheduler
#from torchvision import transforms

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
#from accelerate.state import AcceleratorState

import diffusers
from diffusers.utils import check_min_version, is_wandb_available #, deprecate, make_image_grid
from diffusers.optimization import get_scheduler
#from diffusers.utils.torch_utils import is_compiled_module
#from diffusers import AutoencoderKL

from diffusers.training_utils import EMAModel #,compute_snr
#from diffusers.utils.import_utils import is_xformers_available
#from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card

from diffusers.models.vq_gan_3d import VQGAN #, LPIPS, NLayerDiscriminator, NLayerDiscriminator3D
from diffusers.models.ddpm import PatchUnet3D, PatchGaussianDiffusion
from diffusers.models.ddpm.diffusion import default, is_list_str
from diffusers.models.ddpm.text import tokenize, bert_embed #, BERT_MODEL_DIM
#from diffusers.models.vq_gan_3d.utils import adopt_weight

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")
logger = get_logger(__name__, log_level="INFO")

# def get_model_checksum(model):
#     checksum = 0
#     for param in model.parameters():
#         checksum += param.detach().sum().item()
#     return checksum

# def log_norm_layer_params(model, description=""):
#     """
#     Log the parameters of LayerNorm and GroupNorm layers in the model.
#     Args:
#         model (torch.nn.Module): The model containing the normalization layers.
#         description (str): Description to include in the log for context (e.g., "before saving").
#     """
#     for name, layer in model.named_modules():
#         if isinstance(layer, (nn.LayerNorm, nn.GroupNorm)):
#             logger.info(f"{description} - {name} - weight: {layer.weight.sum().item()}, bias: {layer.bias.sum().item()}")



@torch.no_grad()
def log_validation(input_size, test_dataloader, vae, unet3d, noise_scheduler, accelerator, weight_dtype, gloabal_step, num_samples, save_dir):
    logger.info("Running validation... ")
    vae.eval()
    unet3d.eval()
    vae = accelerator.unwrap_model(vae)
    unet3d = accelerator.unwrap_model(unet3d)

    os.makedirs(os.path.join(save_dir, "generated_volumes"), exist_ok=True)

    images = []
    with torch.autocast(accelerator.device.type, dtype=weight_dtype):
        batch = next(iter(test_dataloader))
        x = batch["pixel_values"][:num_samples]
        cond = batch["condition"][:num_samples]
        patch_position = batch["patch_position"][:num_samples]
        low_res_guidance = batch["lowres_guide"][:num_samples]
        
        for i in range(num_samples):
            
            # patch_index = 13
            # cond = torch.tensor([0.5], dtype=torch.float16).to(unet3d.device)
            # patch_position = torch.tensor(patch_index, dtype=torch.long).unsqueeze(0).to(unet3d.device)
            # lowres_guide = torch.load("/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_w/results/low-res_gen/tensor_low-res_image_origin.pth").to(torch.float64).to(unet3d.device)

            # lowres_guide = F.interpolate(
            #     lowres_guide,
            #     size=(218, 182, 182),
            #     mode='trilinear',
            #     align_corners=False
            # )
            # lowres_guide = lowres_guide.squeeze(0).to(torch.float16)

            # # Initialize lists
            # depth_starts = [0, 71, 142]
            # height_starts = [0, 59, 118]
            # width_starts = [0, 59, 118]

            # # Initialize index
            # index = 0
            # # Mapping list to store results
            # mapping = []

            # for id, start_d in enumerate(depth_starts):
            #     for ih, start_h in enumerate(height_starts):
            #         for iw, start_w in enumerate(width_starts):
            #             mapping.append({
            #                 'index': index,
            #                 'start_d': start_d,
            #                 'start_h': start_h,
            #                 'start_w': start_w
            #             })
            #             print(f"Index: {index}, Start Coordinates: (D: {start_d}, H: {start_h}, W: {start_w})")
            #             index += 1

            # mapping_entry = mapping[patch_index] ###
            # start_d = mapping_entry['start_d']
            # start_h = mapping_entry['start_h']
            # start_w = mapping_entry['start_w']
            
            # # Patch dimensions
            # pd, ph, pw = 76, 64, 64
            
            # # Extract the patch
            # patch = lowres_guide[
            #     :, 
            #     start_d:start_d + pd,
            #     start_h:start_h + ph,
            #     start_w:start_w + pw
            # ]

            # low_res_guidance = F.interpolate(
            #     patch.unsqueeze(0).to(torch.float64),
            #     size=(38, 32, 32),
            #     mode='trilinear',
            #     align_corners=False
            # ).to(torch.float16)
            
            image = noise_scheduler.sample(vae=vae,
                                        unet=unet3d,
                                        image_size=int(input_size[1] / vae.config.downsample[1]),
                                        num_frames=int(input_size[0] / vae.config.downsample[0]),
                                        channels=int(vae.config.embedding_dim),
                                        patch_position=patch_position[i].unsqueeze(0), ###
                                        low_res_guidance=low_res_guidance[i].unsqueeze(0), ###
                                        cond=cond[i].unsqueeze(0), ###
                                        cond_scale=1., ###
                                        batch_size=1
                                        )
            images.append(image)
            image_np = image.squeeze().cpu().numpy()  # Shape: (D, H, W)

            # Save as .npy file
            npy_path = os.path.join(os.path.join(save_dir, "generated_volumes"), f"generated_volume_checkpoint_{gloabal_step}_{i}.npy")
            np.save(npy_path, image_np)
            logger.info(f"Saved generated image to {npy_path}")
    
    images = torch.stack([x.cpu(), torch.cat(images, dim=0).cpu()], dim=1)

    for tracker in accelerator.trackers:
        tracker.name = "wandb"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("Original (left), Reconstruction (right)", np_images, gloabal_step)
        elif tracker.name == "wandb":
            # tracker.log(
            #     {
            #         "validation Sagittal": [
            #             wandb.Image(image[:, :, image.shape[2] // 2, :, :].squeeze(), caption=f"{i}: Uncond generation.")
            #             for i, image in enumerate(images)
            #         ],
            #         "validation Coronal": [
            #             wandb.Image(image[:, :, :, image.shape[3] // 2, :].squeeze(), caption=f"{i}: Uncond generation.")
            #             for i, image in enumerate(images)
            #         ],
            #         "validation Axial": [
            #             wandb.Image(image[:, :, :, :, image.shape[4] // 2].squeeze(), caption=f"{i}: Uncond generation.")
            #             for i, image in enumerate(images)
            #         ]
            #     }
            # )=
            tracker.log(
                {
                    "Original (left), Generation (right) - Sagittal": [
                        wandb.Image(torchvision.utils.make_grid(image[:, :, image.shape[2] // 2, :, :]).permute(1, 2, 0))
                        for _, image in enumerate(images)
                    ],
                    "Original (left), Generation (right) - Coronal": [
                        wandb.Image(torchvision.utils.make_grid(image[:, :, :, image.shape[3] // 2, :]).permute(1, 2, 0))
                        for _, image in enumerate(images)
                    ],
                    "Original (left), Generation (right) - Axial": [
                        wandb.Image(torchvision.utils.make_grid(image[:, :, :, :, image.shape[4] // 2]).permute(1, 2, 0))
                        for _, image in enumerate(images)
                    ]
                }
            )
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    del vae
    del images
    torch.cuda.empty_cache()

def count_activations(model, dummy_input, time, patch_position, cond, low_res_input):
    activations = []
    def hook_fn(module, input, output):
        # Add the number of elements in the output to the activations list
        activations.append(output.numel())

    # Register a forward hook for each module to capture the output (activations)
    hooks = []
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d, 
                              nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.Embedding,)):
                              #nn.ReLU, nn.GELU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU, nn.Parameter
                              #nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            hooks.append(layer.register_forward_hook(hook_fn))
    
    # Forward pass through the model
    with torch.no_grad():
        model(dummy_input, time=time, patch_position=patch_position, low_res_guidance=low_res_input, cond=cond)
    
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
            print(f"Layer {name}: {num_params} parameters (Trainable: {param.requires_grad})")

    return param_count




def parse_args():
    parser = argparse.ArgumentParser(description="UNET3D training script.")
    # parser.add_argument(
    #     "--revision",
    #     type=str,
    #     default=None,
    #     required=False,
    #     help="Revision of pretrained model identifier from huggingface.co/models.",
    # )
    # parser.add_argument(
    #     "--variant",
    #     type=str,
    #     default=None,
    #     help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    # )
    parser.add_argument(
        "--axis",
        type=str,
        default=None,
        help=(
            "3D volume's axis that will be processed as the temporal axis."
        ),
    )
    parser.add_argument(
        "--dim_mults",
        type=str,
        default=None,
        help=(
            "dim_mults for Unet3D."
        ),
    )
    parser.add_argument(
        "--attn_heads",
        type=int,
        default=None,
        help=(
            "attn_heads for Unet3D."
        ),
    )
    parser.add_argument(
        "--pretrained_vae_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained vae or vae identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--loss_type", type=str, default="l1", help="Loss type: l1 or l2."
    )
    parser.add_argument(
        "--num_timesteps", type=int, default=1000, help="The number of diffusion steps."
    )
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
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
    # parser.add_argument(
    #     "--max_train_samples",
    #     type=int,
    #     default=None,
    #     help=(
    #         "For debugging purposes or quicker training, truncate the number of training examples to this "
    #         "value if set."
    #     ),
    # )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    # parser.add_argument(
    #     "--cache_dir",
    #     type=str,
    #     default=None,
    #     help="The directory where the downloaded models and datasets will be stored.",
    # )
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
    # parser.add_argument(
    #     "--center_crop",
    #     default=False,
    #     action="store_true",
    #     help=(
    #         "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
    #         " cropped. The images will be resized to the resolution first before cropping."
    #     ),
    # )
    # parser.add_argument(
    #     "--random_flip",
    #     action="store_true",
    #     help="whether to randomly flip images horizontally",
    # )
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
            ' "constant", "constant_with_warmup", "piecewise_constant"]'
        ),
    )
    # parser.add_argument(
    #     "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.",
    # )
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
    # parser.add_argument(
    #     "--non_ema_revision",
    #     type=str,
    #     default=None,
    #     required=False,
    #     help=(
    #         "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
    #         " remote repository specified with --pretrained_model_name_or_path."
    #     ),
    # )
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
    # parser.add_argument(
    #     "--validation_epochs",
    #     type=int,
    #     default=5,
    #     help="Run validation every X epochs.",
    # )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="vae-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument("--num_samples", type=int, default=5, help="The number of samples for validation.")
    # parser.add_argument(
    #     "--kl_scale",
    #     type=float,
    #     default=1e-6,
    #     help="Scaling factor for the Kullback-Leibler divergence penalty term.",
    # )
    # parser.add_argument(
    #     "--lpips_scale",
    #     type=float,
    #     default=5e-1,
    #     help="Scaling factor for the LPIPS metric",
    # )
    # parser.add_argument(
    #     "--lpips_start",
    #     type=int,
    #     default=50001,
    #     help="Start for the LPIPS metric",
    # )
    # parser.add_argument(
    #     "--tile_sample_size",
    #     type=int,
    #     default=None,
    #     help="Start for the LPIPS metric",
    # )
    # parser.add_argument(
    #     "--slicing",
    #     action="store_true",
    #     help="Enable sliced VAE (process single batch at a time)",
    # )
    # parser.add_argument(
    #     "--tiling",
    #     action="store_true",
    #     help="Enable tiling VAE (process divided image)",
    # )    
    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")
    
    # default to using the same revision for the non-ema model if not specified
    # if args.non_ema_revision is None:
    #     args.non_ema_revision = args.revision

    return args



# Define custom dataset class
class UKB_Dataset(Dataset):
    """
    A PyTorch Dataset for loading UKB images and labels.
    
    Args:
        image_dir (str): Directory containing image files.
        label_dir (str): CSV file containing image IDs and labels.
        transform (callable, optional): Optional transform to apply to samples.
        axis (str): Axis to permute ('a', 'c', or 's').
    """
    def __init__(self, image_dir, label_dir, transform=None, axis="s", patch_position=0):
        super().__init__()
        self.data_dir = image_dir
        data_csv = pd.read_csv(label_dir)
        self.image_names = list(data_csv['id'])

        # load conditioning variable: age
        self.ages = data_csv['age'].values.astype(np.float16)
        self.min_age, self.max_age = self.ages.min(), self.ages.max()
        self.norm_ages = (self.ages - self.min_age) / (self.max_age - self.min_age)

        # # load conditioning variable: ventricular volume
        # self.bvvs = data_csv['ventricular_volume'].values.astype(np.float32)
        # self.min_bvv, self.max_bvv = self.bvvs.min(), self.bvvs.max()
        # self.norm_bvvs = (self.bvvs - self.min_bvv) / (self.max_bvv - self.min_bvv)

        # # load conditioning variable: gender
        # self.genders = data_csv['gender'].values
        # self.gender_encoded = []
        # for gender_str in self.genders:
        #     if gender_str == 'Male':
        #         self.gender_encoded.append([1.0, 0.0])
        #     elif gender_str == 'Female':
        #         self.gender_encoded.append([0.0, 1.0])
        #     else:
        #         self.gender_encoded.append([0.0, 0.0])  # Handle unknown gender
        # self.gender_encoded = np.array(self.gender_encoded, dtype=np.float16)

        self.patch_position = patch_position
        self.transform = transform
        self.axis = axis
        self.image_paths = [
            os.path.join(self.data_dir, f'final_array_128_full_{name}.npy')
            for name in self.image_names
        ]

        # Initialize lists
        depth_starts = [0, 71, 142]
        height_starts = [0, 59, 118]
        width_starts = [0, 59, 118]

        # Initialize index
        index = 0
        # Mapping list to store results
        self.mapping = []

        for id, start_d in enumerate(depth_starts):
            for ih, start_h in enumerate(height_starts):
                for iw, start_w in enumerate(width_starts):
                    self.mapping.append({
                        'index': index,
                        'start_d': start_d,
                        'start_h': start_h,
                        'start_w': start_w
                    })
                    print(f"Index: {index}, Start Coordinates: (D: {start_d}, H: {start_h}, W: {start_w})")
                    index += 1

        # low-res guidance
        #self.lowres_guidance = torch.load("/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_w/results/low-res_gen/tensor_low-res_image_origin.pth").to(torch.float64)

    def get_patch_by_index(self, volume, index, mapping):
        # Retrieve the start coordinates from the mapping
        mapping_entry = mapping[index]
        start_d = mapping_entry['start_d']
        start_h = mapping_entry['start_h']
        start_w = mapping_entry['start_w']
        
        # Patch dimensions
        pd, ph, pw = 76, 64, 64
        
        # Extract the patch
        patch = volume[
            :, 
            start_d:start_d + pd,
            start_h:start_h + ph,
            start_w:start_w + pw
        ]
        return patch

    def gaussian_kernel_3d(self, kernel_size=5, sigma=2.0, device='cpu'):
        """
        Create a 3D Gaussian kernel for convolution.
        kernel_size: int (odd number), size of the kernel for D,H,W.
        sigma: float, standard deviation for Gaussian.
        """
        coords = torch.arange(kernel_size, dtype=torch.float32, device=device)
        coords = coords - (kernel_size - 1) / 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        
        # g is 1D, make it 3D by outer products
        g_3d = g[:, None, None] * g[None, :, None] * g[None, None, :]
        g_3d = g_3d / g_3d.sum()
        
        # Shape (1,1,D,H,W)
        g_3d = g_3d.unsqueeze(0).unsqueeze(0)
        return g_3d

    def gaussian_blur_3d(self, volume, kernel_size=5, sigma=2.0):
        """
        Apply 3D Gaussian blur to a volume of shape (1, D, H, W).
        
        Args:
            volume (torch.Tensor): shape (1, D, H, W), single-channel volume.
            kernel_size (int): Kernel size for Gaussian.
            sigma (float): Standard deviation for Gaussian.
            
        Returns:
            torch.Tensor: Blurred volume of shape (1, D, H, W).
        """
        device = volume.device
        C = volume.shape[0]
        assert C == 1, "This function assumes a single channel (C=1)."
        assert volume.ndim == 4, "Volume should have shape (1, D, H, W)."
        
        kernel = self.gaussian_kernel_3d(kernel_size, sigma, device=device)  # (1,1,D,H,W)
        
        # Add a batch dimension: (1, C=1, D, H, W)
        volume = volume.unsqueeze(0)  # shape now (1,1,D,H,W)
        
        # Perform convolution with padding to maintain shape
        padding = kernel_size // 2
        blurred = F.conv3d(volume, kernel, padding=padding, groups=1)
        # blurred shape is still (1,1,D,H,W)
        
        # Remove batch dimension
        blurred = blurred.squeeze(0)  # back to (1, D, H, W)
        return blurred

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        #lowres_guide = self.lowres_guidance

        try:
            image = np.load(image_path)  # (128,128,128,1)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = torch.from_numpy(image)  # Stays on CPU
        
        axes_mapping = {
            's': (3, 0, 1, 2),
            'c': (3, 1, 0, 2),
            'a': (3, 2, 1, 0)
        }

        try:
            image = image.permute(*axes_mapping[self.axis])  # (1,128,128,128)
        except KeyError:
            raise ValueError("axis must be one of 'a', 'c', or 's'.")
        
        # Add batch dimension (N=1) for interpolation
        image = image.unsqueeze(0)  # Shape: (1, 1, D, H, W)

        # Define target size
        target_size = (218, 182, 182)  # (D₂, H₂, W₂)
        lowres_size = (76, 64, 64)

        # Resize the volume using trilinear interpolation
        image = F.interpolate(
            image,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )  # Shape: (1, 1, D₂, H₂, W₂)
        lowres_guide = F.interpolate(
            image,
            size=lowres_size,
            mode='trilinear',
            align_corners=False
        )  # Shape: (1, 1, D₂, H₂, W₂)
        lowres_guide = F.interpolate(
            lowres_guide,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )  # Shape: (1, 1, D₂, H₂, W₂)

        # Remove the batch dimension
        image = image.squeeze(0).to(torch.float16)  # Shape: (1, D₂, H₂, W₂) # (1,218,182,182)
        lowres_guide = lowres_guide.squeeze(0).to(torch.float16)  # Shape: (1, D₂, H₂, W₂) # (1,218,182,182)

        age = torch.tensor([self.norm_ages[index]], dtype=torch.float16) # Shape: [1]
        # gender = torch.tensor(self.gender_encoded[index], dtype=torch.float16)  # Shape: [2]
        # bvv = torch.tensor([self.normalized_bvvs[index]], dtype=torch.float16)  # Shape: [1]

        #cond_tensor = torch.cat([age, gender, bvv], dim=-1)  # Shape: [4]

        sample = {
            "pixel_values": image,
            "condition": age,
            "lowres_guide": lowres_guide,
        }

        if self.transform:
            sample = self.transform(sample)
        del image

        patch_position_sampled = random.randint(0, 26)
        sample["pixel_values"] = self.get_patch_by_index(sample["pixel_values"], patch_position_sampled, self.mapping)
        sample["lowres_guide"] = self.get_patch_by_index(sample["lowres_guide"], patch_position_sampled, self.mapping)
        sample["lowres_guide"] = self.gaussian_blur_3d(sample["lowres_guide"])
        sample["patch_position"] = torch.tensor(patch_position_sampled, dtype=torch.long)

        return sample



def main():
    args = parse_args()
    input_size = tuple(int(x) for x in args.resolution.split(","))

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    # if args.non_ema_revision is not None:
    #     deprecate(
    #         "non_ema_revision!=None",
    #         "0.15.0",
    #         message=(
    #             "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
    #             " use `--variant=non_ema` instead."
    #         ),
    #     )
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
        # if args.push_to_hub:
        #     repo_id = create_repo(
        #         repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
        #     ).repo_id

    # Load VAE & Unet3D
    vae = VQGAN.from_pretrained( ### set to pretrained VQGAN model path
        args.pretrained_vae_path, subfolder="vqgan", 
    ).to(accelerator.device)
    vae.requires_grad_(False)

    dim_mults = tuple(int(x) for x in args.dim_mults.split(","))
    unet3d = PatchUnet3D(
        dim=int(input_size[1] / vae.config.downsample[1]), # 32
        cond_dim=1, # pheno dim (patch dim added internally by model)
        dim_mults=dim_mults, # "1,2,4,8,16"
        channels=int(vae.config.embedding_dim), # 8
        attn_heads=args.attn_heads, # 24
        attn_dim_head=int(args.attn_heads*2), # 48
        num_patch_positions=27, ###
        patch_position_embedding_dim=16, ###
        low_res_guidance_channel=1, ###
        guidance_dim=256, ###
        #guidance_embedding_dim=128, ###
    ).to(accelerator.device)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet3d = PatchUnet3D(
            dim=int(input_size[1] / vae.config.downsample[1]), # 32
            cond_dim=1, # pheno dim (patch dim added internally by model)
            dim_mults=dim_mults, # "1,2,4,8,16"
            channels=int(vae.config.embedding_dim), # 8
            attn_heads=args.attn_heads, # 24
            attn_dim_head=int(args.attn_heads*2), # 48
            num_patch_positions=27, ###
            patch_position_embedding_dim=16, ###
            low_res_guidance_channel=1, ###
            guidance_dim=256, ###
            #guidance_embedding_dim=128, ###
        )
        ema_unet3d = EMAModel(
            ema_unet3d.parameters(), 
            decay=0.995,
            update_after_step=2000,
            model_cls=PatchUnet3D, 
            model_config=ema_unet3d.config
        )

    noise_scheduler = PatchGaussianDiffusion( # diffusers pipeline?
        #unet3d,
        #vqgan_ckpt=cfg.model.vqgan_ckpt,
        #image_size=cfg.model.diffusion_img_size,
        #num_frames=cfg.model.diffusion_depth_size,
        #channels=cfg.model.diffusion_num_channels,
        timesteps=1000,
        # sampling_timesteps=cfg.model.sampling_timesteps,
        #loss_type=cfg.model.loss_type,
        # objective=cfg.objective
    ).to(accelerator.device) ###args

    # vae.train()
    # image_discriminator.train()
    # video_discriminator.train()
    # config = OmegaConf.load(args.vqgan_config)
    # model = VQGAN(config)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet3d.save_pretrained(os.path.join(output_dir, "unet3d_ema"))

                #logger.info(f"{unet3d = }") # print model architecture
                for _, model in enumerate(models):
                    if isinstance(model, type(accelerator.unwrap_model(unet3d))):
                        model.save_pretrained(os.path.join(output_dir, "unet3d"))
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")
                    
                    # make sure to pop weight so that corresponding model is not saved again
                    if weights: ### https://github.com/huggingface/diffusers/issues/2606#issuecomment-1704077101
                        weights.pop()

                # vae = vae[0]
                # vae.save_pretrained(os.path.join(output_dir, "vae"))
                # weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet3d_ema"), PatchUnet3D)
                ema_unet3d.load_state_dict(load_model.state_dict())
                ema_unet3d.to(accelerator.device)
                del load_model

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                if isinstance(model, type(accelerator.unwrap_model(unet3d))):
                    load_model = PatchUnet3D.from_pretrained(input_dir, subfolder="unet3d")
                else:
                    raise ValueError(f"unexpected load model: {model.__class__}")

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

    # if args.gradient_checkpointing:
    #     vae.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

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
        optimizer_cls = bnb.optim.Adam8bit #AdamW8bit
    else:
        optimizer_cls = torch.optim.Adam #AdamW

    # optimizer = optimizer_cls(
    #     vae.parameters(),
    #     lr=args.learning_rate,
    #     betas=(args.adam_beta1, args.adam_beta2),
    #     weight_decay=args.adam_weight_decay,
    #     eps=args.adam_epsilon,
    # )
    optimizer = optimizer_cls(unet3d.parameters(), lr=args.learning_rate)

    # Load data
    train_transforms = transforms.Compose(
        [
            transforms.ScaleIntensityd(keys=["pixel_values"], minv=-1.0, maxv=1.0),
            transforms.ThresholdIntensityd(keys=["pixel_values"], threshold=1, above=False, cval=1.0),
            transforms.ThresholdIntensityd(keys=["pixel_values"], threshold=-1, above=True, cval=-1.0),
            transforms.ToTensord(keys=["pixel_values"]),
            transforms.ScaleIntensityd(keys=["lowres_guide"], minv=-1.0, maxv=1.0),
            transforms.ThresholdIntensityd(keys=["lowres_guide"], threshold=1, above=False, cval=1.0),
            transforms.ThresholdIntensityd(keys=["lowres_guide"], threshold=-1, above=True, cval=-1.0),
            transforms.ToTensord(keys=["lowres_guide"]),
        ]
    )

    with accelerator.main_process_first():
        if args.data_dir is not None: # args.test_data_dir is not None and args.data_dir is not None:
            train_dataset = UKB_Dataset(args.data_dir, args.train_label_dir, transform=train_transforms, axis=args.axis)
            valid_dataset = UKB_Dataset(args.data_dir, args.valid_label_dir, transform=train_transforms, axis=args.axis)

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

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # lr_scheduler = get_scheduler(
    #     args.lr_scheduler,
    #     optimizer=optimizer,
    # )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        power=3.0
    )

    # Prepare everything with our `accelerator`.
    unet3d, optimizer, lr_scheduler, train_dataloader, test_dataloader = accelerator.prepare(
        unet3d, optimizer, lr_scheduler, train_dataloader, test_dataloader
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

    # Move vae (and ema model) to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    if args.use_ema:
        ema_unet3d.to(accelerator.device, dtype=weight_dtype)

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
    # def unwrap_model(model):
    #     model = accelerator.unwrap_model(model)
    #     model = model._orig_mod if is_compiled_module(model) else model
    #     return model
        
   


    # ------------------------------ TRAIN ------------------------------ #
    #args.num_timesteps = 1000
    #args.loss_type = "l1"
    # args.image_gan_weight = 1.0
    # args.video_gan_weight = 1.0
    # args.l1_weight = 4.0
    # args.gan_feat_weight = 4.0
    # args.perceptual_weight = 4.0

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
            # # Log immediately after loading state
            # print("################################")
            # # for param_id, state in optimizer.state.items():
            # #     logger.info(f"after loading - Param - exp_avg: {state['exp_avg'].sum().item()}")
            # #     logger.info(f"after loading - Param - exp_avg_sq: {state['exp_avg_sq'].sum().item()}")
            # for i, param_group in enumerate(optimizer.param_groups):
            #     print(param_group.keys())
            #     logger.info(f"Optimizer state for param group {i} after save checkpoint-{global_step} betas: {param_group['betas']}")
            #     logger.info(f"Optimizer state for param group {i} after save checkpoint-{global_step} lr: {param_group['lr']}")
            # model_checksum_after_load = get_model_checksum(unet3d)
            # logger.info(f"Model checksum after loading checkpoint-{global_step}: {model_checksum_after_load}")
            # log_norm_layer_params(unet3d, "After loading")
            # print("################################")
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

    # Model memory check
    # dummy_input = torch.ones(1, 8, 38, 32, 32).float().to(accelerator.device) # 262,144 for 64^3
    # low_res_input = torch.ones(1, 1, 38, 32, 32).float().to(accelerator.device)
    # timesteps = torch.randint(0, args.num_timesteps, (dummy_input.shape[0],)).long().to(accelerator.device)
    # cond = torch.tensor([0.5]).float().to(accelerator.device)
    # patch_position = torch.tensor(13, dtype=torch.long).unsqueeze(0).to(accelerator.device)
    # unet3d.eval()

    #unet3d_activations = count_activations(unet3d, dummy_input, timesteps, patch_position, cond, low_res_input)
    #print(f"### UNET3D's number of activations: {unet3d_activations:,}")
    
    # Show model architecture and # of params
    # param_counts = count_parameters(unet3d)
    # print(f"### UNET3D's Total parameters: {param_counts['total']}")
    # print(f"### UNET3D's Trainable parameters: {param_counts['trainable']}")
    # print(f"### UNET3D's Non-trainable parameters: {param_counts['non_trainable']}")
    # print(f"### UNET3D's Parameters by layer type: {param_counts['by_layer_type']}")
    # summary(unet3d, (8, 32, 32, 32))
    # summary(unet3d, (8, 32, 32, 32, 32)) # assertion
    
    # Training
    if accelerator.is_main_process:
        print("### Start training ###")
    for epoch in range(first_epoch, args.num_train_epochs):
        
        logger.info(f"{epoch = }")
        for step, batch in enumerate(train_dataloader):
            start_time = time.time()
            unet3d.train()

            # Accumulate gradients
            with accelerator.accumulate(unet3d):
                with torch.autocast(accelerator.device.type, dtype=weight_dtype):
                    x = batch["pixel_values"].to(weight_dtype)
                    
                    with torch.no_grad():
                        z = vae.encode(x, quantize=False, include_embeddings=True)
                        # normalize to -1 and 1 (apply scaling factor)
                        latents = ((z - vae.codebook.embeddings.min()) /
                                   (vae.codebook.embeddings.max() -
                                    vae.codebook.embeddings.min())) * 2.0 - 1.0
                    
                    print("### latents.shape:", latents.shape, flush=True) # torch.Size([10, 8, 32, 32, 32])
                    B = latents.shape[0]
                    check_shape(latents, 'b c f h w', 
                                c=int(vae.config.embedding_dim), 
                                f=int(input_size[0] / vae.config.downsample[0]), # saggital
                                h=int(input_size[1] / vae.config.downsample[1]), # coronal
                                w=int(input_size[2] / vae.config.downsample[2])) # axial
                    
                    timesteps = torch.randint(0, args.num_timesteps, (B,), device=latents.device).long()

                    noise = torch.randn_like(latents)
                    if args.noise_offset:
                        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                        noise += args.noise_offset * torch.randn(
                            (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                        )
                    if args.input_perturbation:
                        noise += args.input_perturbation * torch.randn_like(noise)

                    noisy_latents = noise_scheduler.q_sample(x_start=latents, t=timesteps, noise=noise)

                    # conditioning
                    cond = batch["condition"].to(latents.device)
                    if is_list_str(cond):
                        cond = bert_embed(tokenize(cond), return_cls_repr=False)
                        cond = cond.to(latents.device)

                    z_pred = unet3d(x=noisy_latents, 
                                    time=timesteps, 
                                    patch_position=batch["patch_position"], 
                                    low_res_guidance=batch["lowres_guide"],
                                    cond=cond, 
                                    null_cond_prob=0.1)

                    if args.loss_type == 'l1':
                        loss = F.l1_loss(noise, z_pred)
                    elif args.loss_type == 'l2':
                        loss = F.mse_loss(noise, z_pred)
                    else:
                        raise NotImplementedError()

                    if not torch.isfinite(loss):
                        logger.info("\nWARNING: non-finite loss.")
                        continue
                        
                    optimizer.zero_grad(set_to_none=True)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(unet3d.parameters(), args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()

                    gathered_train_loss = accelerator.gather(loss)
                    step_train_loss = gathered_train_loss.sum()

                # if step % 50 == 0:
                logs = {
                    "unet_step_train_loss": step_train_loss.detach().item() / B,
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                accelerator.log(logs)

            end_time = time.time()
            if accelerator.is_main_process:
                print(f"### Training step elasped: {end_time - start_time}", flush=True)

            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet3d.step(unet3d.parameters())
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

                    # Log before saving state
                    # print("################################")
                    # # for param_id, state in optimizer.state.items():
                    # #     logger.info(f"before saving - Param - exp_avg: {state['exp_avg'].sum().item()}")
                    # #     logger.info(f"before saving - Param - exp_avg_sq: {state['exp_avg_sq'].sum().item()}")
                    # for i, param_group in enumerate(optimizer.param_groups):
                    #     print(param_group.keys())
                    #     logger.info(f"Optimizer state for param group {i} after save checkpoint-{global_step} betas: {param_group['betas']}")
                    #     logger.info(f"Optimizer state for param group {i} after save checkpoint-{global_step} lr: {param_group['lr']}")
                    # model_checksum_before_save = get_model_checksum(unet3d)
                    # logger.info(f"Model checksum before saving checkpoint-{global_step}: {model_checksum_before_save}")
                    # log_norm_layer_params(unet3d, "Before saving")
                    # print("################################")

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                    if args.use_ema:
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        ema_unet3d.store(unet3d.parameters())
                        ema_unet3d.copy_to(unet3d.parameters())

                    # validation
                    if accelerator.is_main_process:
                        print("### Start validation ###")
                        log_validation(input_size, test_dataloader, vae, unet3d, noise_scheduler, accelerator, weight_dtype, global_step, args.num_samples, save_path)
                    
                    with torch.no_grad():
                        vae.eval()
                        unet3d.eval()
                        valid_loss = 0.0
                        for step, batch in enumerate(test_dataloader):
                            with torch.autocast(accelerator.device.type, dtype=weight_dtype):
                                x = batch["pixel_values"].to(weight_dtype)

                                z = vae.encode(x, quantize=False, include_embeddings=True)
                                # normalize to -1 and 1 (apply scaling factor)
                                latents = ((z - vae.codebook.embeddings.min()) /
                                           (vae.codebook.embeddings.max() -
                                            vae.codebook.embeddings.min())) * 2.0 - 1.0
                                
                                B = latents.shape[0]
                                check_shape(latents, 'b c f h w', 
                                            c=int(vae.config.embedding_dim), 
                                            f=int(input_size[0] / vae.config.downsample[0]), # saggital
                                            h=int(input_size[1] / vae.config.downsample[1]), # coronal
                                            w=int(input_size[2] / vae.config.downsample[2])) # axial
                                
                                timesteps = torch.randint(0, args.num_timesteps, (B,), device=latents.device).long()

                                noise = torch.randn_like(latents)
                                if args.noise_offset:
                                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                                    noise += args.noise_offset * torch.randn(
                                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                                    )
                                if args.input_perturbation:
                                    noise += args.input_perturbation * torch.randn_like(noise)

                                noisy_latents = noise_scheduler.q_sample(x_start=latents, t=timesteps, noise=noise)

                                # conditioning
                                cond = batch["condition"].to(latents.device)
                                if is_list_str(cond):
                                    cond = bert_embed(tokenize(cond), return_cls_repr=False)
                                    cond = cond.to(latents.device)

                                z_pred = unet3d(x=noisy_latents, 
                                                time=timesteps, 
                                                patch_position=batch["patch_position"], 
                                                low_res_guidance=batch["lowres_guide"],
                                                cond=cond)

                                if args.loss_type == 'l1':
                                    val_loss = F.l1_loss(noise, z_pred)
                                elif args.loss_type == 'l2':
                                    val_loss = F.mse_loss(noise, z_pred)
                                else:
                                    raise NotImplementedError()

                                if not torch.isfinite(val_loss):
                                    logger.info("\nWARNING: non-finite loss.")
                                    continue

                                # Gather the losses across all processes for logging (if we use distributed training).
                                gathered_valid_loss = accelerator.gather(val_loss)
                                valid_loss += gathered_valid_loss.sum()
                                #valid_loss += val_loss.sum()
                                
                                # val_avg_loss = accelerator.gather(val_loss.repeat(args.valid_batch_size)).mean()
                                # valid_loss += val_avg_loss.item() # / args.gradient_accumulation_steps

                            # if accelerator.sync_gradients:
                            #     #accelerator.log({"valid_loss": valid_loss}, step=global_step)
                            #     if accelerator.is_main_process:
                            #         print("### valid loss:", valid_loss)
                            #     valid_loss = 0.0

                        logs = {
                            #"valid_step_loss": val_loss.detach().item(),
                            "valid_loss": valid_loss.detach().item() / len(test_dataloader.dataset),
                        }
                        accelerator.log(logs)

                    if args.use_ema:
                    # Switch back to the original UNet parameters.
                        ema_unet3d.restore(unet3d.parameters())

            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    # if accelerator.is_main_process:
    #     vae = unwrap_model(vae) # accelerator.unwrap_model(vae)
    #     vae.save_pretrained(args.output_dir)
    accelerator.end_training()

if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    main()