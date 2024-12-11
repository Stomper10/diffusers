import os
import time
import math
import shutil
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
from diffusers.models.ddpm import Unet3D, GaussianDiffusion
from diffusers.models.ddpm.diffusion import default, is_list_str
from diffusers.models.ddpm.text import tokenize, bert_embed #, BERT_MODEL_DIM
#from diffusers.models.vq_gan_3d.utils import adopt_weight

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")
logger = get_logger(__name__, log_level="INFO")

@torch.no_grad()
def log_validation(input_size, test_dataloader, vae, unet3d, noise_scheduler, accelerator, weight_dtype, gloabal_step, num_samples, save_dir):
    logger.info("Running validation... ")
    vae = accelerator.unwrap_model(vae)
    unet3d = accelerator.unwrap_model(unet3d)

    os.makedirs(os.path.join(save_dir, "generated_volumes"), exist_ok=True)

    images = []
    with torch.autocast(accelerator.device.type, dtype=weight_dtype):
        for i in range(num_samples):
            
            age = torch.tensor([1.0], dtype=torch.float16).to(unet3d.device)

            image = noise_scheduler.sample(vae=vae,
                                        unet=unet3d,
                                        image_size=int(input_size[1] / vae.config.downsample[1]),
                                        num_frames=int(input_size[0] / vae.config.downsample[0]),
                                        channels=int(vae.config.embedding_dim),
                                        cond=age, #####
                                        batch_size=1
                                        )
            images.append(image)
            image_np = image.squeeze().cpu().numpy()  # Shape: (D, H, W)

            # Save as .npy file
            npy_path = os.path.join(os.path.join(save_dir, "generated_volumes"), f"generated_volume_checkpoint_{gloabal_step}_{i}.npy")
            np.save(npy_path, image_np)
            logger.info(f"Saved generated image to {npy_path}")

        x = next(iter(test_dataloader))["pixel_values"][:num_samples]
    
    images = torch.stack([x.cpu(), torch.cat(images, dim=0).cpu()], dim=1)

    del vae
    del images
    torch.cuda.empty_cache()



def parse_args():
    parser = argparse.ArgumentParser(description="Whole MRI (low-res & low-qual) generation script.")
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank



def main():
    args = parse_args()
    input_size = tuple(int(x) for x in args.resolution.split(","))

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    vae = VQGAN.from_pretrained( ### set to pretrained VQGAN model path
        args.pretrained_vae_path, subfolder="vqgan", 
    ).to(accelerator.device)
    vae.requires_grad_(False)

    unet3d = EMAModel.from_pretrained(
        os.path.join(args.pretrained_unet_path, "unet3d_ema"), Unet3D)
    unet3d.requires_grad_(False)

    noise_scheduler = GaussianDiffusion( # diffusers pipeline?
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

    logger.info("Running generation... ")
    vae = accelerator.unwrap_model(vae)
    unet3d = accelerator.unwrap_model(unet3d)

    os.makedirs(os.path.join(save_dir, "generated_volumes"), exist_ok=True)

    images = []
    with torch.autocast(accelerator.device.type, dtype=weight_dtype):
        for i in range(num_samples):
            
            age = torch.tensor([0.5], dtype=torch.float16).to(unet3d.device)

            image = noise_scheduler.sample(vae=vae,
                                        unet=unet3d,
                                        image_size=int(input_size[1] / vae.config.downsample[1]),
                                        num_frames=int(input_size[0] / vae.config.downsample[0]),
                                        channels=int(vae.config.embedding_dim),
                                        cond=age, #####
                                        batch_size=1
                                        )
            images.append(image)
            image_np = image.squeeze().cpu().numpy()  # Shape: (D, H, W)

            # Save as .npy file
            npy_path = os.path.join(os.path.join(save_dir, "generated_volumes"), f"generated_volume_checkpoint_{gloabal_step}_{i}.npy")
            np.save(npy_path, image_np)
            logger.info(f"Saved generated image to {npy_path}")

        x = next(iter(test_dataloader))["pixel_values"][:num_samples]
    
    images = torch.stack([x.cpu(), torch.cat(images, dim=0).cpu()], dim=1)

    del vae
    del images
    torch.cuda.empty_cache()