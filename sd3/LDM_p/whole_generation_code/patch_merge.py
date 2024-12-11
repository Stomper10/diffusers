import os
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import accelerate
from accelerate import Accelerator

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from monai import transforms

from diffusers.models.vq_gan_3d import VQGAN
from diffusers.models.ddpm import PatchUnet3D, PatchGaussianDiffusion
#from mayavi import mlab

import numpy as np
import matplotlib.pyplot as plt
# import ipywidgets as widgets
# from ipywidgets import interact
# from sklearn.manifold import TSNE

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

depth_starts = [0, 71, 142]
height_starts = [0, 59, 118]
width_starts = [0, 59, 118]

mapping = []
index = 0
for start_d in depth_starts:
    for start_h in height_starts:
        for start_w in width_starts:
            mapping.append({
                'index': index,
                'start_d': start_d,
                'start_h': start_h,
                'start_w': start_w
            })
            index += 1

def merge_patches(patches, mapping, volume_shape):
    """
    Merges a list of patches into a single volume.
    
    Args:
        patches (list of tensors): List of patch tensors with shape (C, D, H, W).
        mapping (list of dicts): List of mappings with 'start_d', 'start_h', 'start_w'.
        volume_shape (tuple): Shape of the whole volume (C, D, H, W).
    
    Returns:
        Tensor: Merged volume tensor with shape volume_shape.
    """
    channels, depth, height, width = volume_shape
    whole_volume = torch.zeros((channels, depth, height, width), dtype=patches[0].dtype)
    count_volume = torch.zeros((channels, depth, height, width), dtype=torch.int)
    
    for idx, patch in enumerate(patches):
        mapping_entry = mapping[idx]
        start_d = mapping_entry['start_d']
        start_h = mapping_entry['start_h']
        start_w = mapping_entry['start_w']
        
        pd, ph, pw = patch.shape[1:]
        
        # Add the patch to the whole volume
        whole_volume[:, start_d:start_d+pd, start_h:start_h+ph, start_w:start_w+pw] += patch
        count_volume[:, start_d:start_d+pd, start_h:start_h+ph, start_w:start_w+pw] += 1
    
    # Handle overlaps by averaging
    count_volume[count_volume == 0] = 1
    whole_volume /= count_volume
    
    return whole_volume

def save_slices_as_images(volume_data):
    slice_images = []
    for i in range(volume_data.shape[0]): ### c:0, s:1, a:2
        plt.figure(figsize=(7, 7))
        plt.imshow(volume_data[i, :, :], cmap='gray') ### c, s, a
        plt.title(f'Slice {i}')
        plt.axis('off')
        
        # Save each slice to a temporary file
        file_name = os.path.join(temp_dir, f'slice_{i}.png')
        plt.savefig(file_name)
        plt.close()
        
        # Open the image and append to the list
        slice_images.append(Image.open(file_name))
    
    return slice_images


# # raw image
# raw_image = np.load("/shared/s1/lab06/20252_individual_samples/final_array_128_full_7075.npy") #.squeeze(0) # (1,76,64,64)
# image = torch.from_numpy(raw_image)  # Stays on CPU
# axes_mapping = {
#     's': (3, 0, 1, 2),
#     'c': (3, 1, 0, 2),
#     'a': (3, 2, 1, 0)
# }

# try:
#     image = image.permute(*axes_mapping["c"])  # (1,128,128,128)
# except KeyError:
#     raise ValueError("axis must be one of 'a', 'c', or 's'.")

# image = image.unsqueeze(0)
# target_size = (218, 182, 182)  # (D₂, H₂, W₂)
# lowres_size = (76, 64, 64)

# # Resize the volume using trilinear interpolation
# image_raw = F.interpolate(
#     image,
#     size=target_size,
#     mode='trilinear',
#     align_corners=False
# )  # Shape: (1, 1, D₂, H₂, W₂)
# lowres_guide = F.interpolate(
#     image_raw,
#     size=lowres_size,
#     mode='trilinear',
#     align_corners=False
# )  # Shape: (1, 1, D₂, H₂, W₂)
# lowres_guide = F.interpolate(
#     lowres_guide,
#     size=target_size,
#     mode='trilinear',
#     align_corners=False
# ).squeeze(0).to(torch.float16)  # Shape: (1, 1, D₂, H₂, W₂)

# temp_dir = "/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/raw_tmp_slices"
# os.makedirs(temp_dir, exist_ok=True)

# raw = save_slices_as_images(image_raw.squeeze(0).to(torch.float16).cpu().squeeze().numpy())
# raw[0].save("/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/raw_1.gif", save_all=True, append_images=raw[1:], duration=200, loop=0)
# low = save_slices_as_images(lowres_guide.cpu().squeeze().numpy())
# low[0].save("/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/low_resolution_guidance_1.gif", save_all=True, append_images=low[1:], duration=200, loop=0)

# for img_file in os.listdir(temp_dir):
#     os.remove(os.path.join(temp_dir, img_file))
# os.rmdir(temp_dir)

# raw_image = np.load("/shared/s1/lab06/20252_individual_samples/final_array_128_full_3966.npy") #.squeeze(0) # (1,76,64,64)
# image = torch.from_numpy(raw_image)  # Stays on CPU
# axes_mapping = {
#     's': (3, 0, 1, 2),
#     'c': (3, 1, 0, 2),
#     'a': (3, 2, 1, 0)
# }

# try:
#     image = image.permute(*axes_mapping["c"])  # (1,128,128,128)
# except KeyError:
#     raise ValueError("axis must be one of 'a', 'c', or 's'.")

# image = image.unsqueeze(0)
# target_size = (218, 182, 182)  # (D₂, H₂, W₂)

# # Resize the volume using trilinear interpolation
# image_raw = F.interpolate(
#     image,
#     size=target_size,
#     mode='trilinear',
#     align_corners=False
# )  # Shape: (1, 1, D₂, H₂, W₂)
# lowres_guide = F.interpolate(
#     image_raw,
#     size=lowres_size,
#     mode='trilinear',
#     align_corners=False
# )  # Shape: (1, 1, D₂, H₂, W₂)
# lowres_guide = F.interpolate(
#     lowres_guide,
#     size=target_size,
#     mode='trilinear',
#     align_corners=False
# ).squeeze(0).to(torch.float16)  # Shape: (1, 1, D₂, H₂, W₂)

# temp_dir = "/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/raw_tmp_slices"
# os.makedirs(temp_dir, exist_ok=True)

# raw = save_slices_as_images(image_raw.squeeze(0).to(torch.float16).cpu().squeeze().numpy())
# raw[0].save("/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/raw_01.gif", save_all=True, append_images=raw[1:], duration=200, loop=0)
# low = save_slices_as_images(lowres_guide.cpu().squeeze().numpy())
# low[0].save("/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/low_resolution_guidance_01.gif", save_all=True, append_images=low[1:], duration=200, loop=0)

# for img_file in os.listdir(temp_dir):
#     os.remove(os.path.join(temp_dir, img_file))
# os.rmdir(temp_dir)



# # low-res guidance
# #low_res_guidance = torch.load("/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_w/results/low-res_gen/tensor_low-res_image_origin.pth").squeeze(0) # (1,76,64,64)
# temp_dir = "/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/lowres_tmp_slices"
# os.makedirs(temp_dir, exist_ok=True)

# lowres = save_slices_as_images(low_res_guidance.cpu().squeeze().numpy())
# lowres[0].save("/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/low_res_guidance.gif", save_all=True, append_images=lowres[1:], duration=200, loop=0)

# for img_file in os.listdir(temp_dir):
#     os.remove(os.path.join(temp_dir, img_file))
# os.rmdir(temp_dir)


# Concat
concat_patches1 = []
concat_patches01 = []

for idx in range(27):
    # Generate or load the patch corresponding to index idx
    # For example, use your model to generate the patch
    concat_patch1 = torch.load(f"/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/patch_generation/concat_patch_{idx}_gen1_tensor.pth").squeeze(0)
    concat_patch01 = torch.load(f"/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/patch_generation/concat_patch_{idx}_gen01_tensor.pth").squeeze(0)
    concat_patches1.append(concat_patch1)
    concat_patches01.append(concat_patch01)

volume_shape = (1, 218, 182, 182)
concat_merged_volume1 = merge_patches(concat_patches1, mapping, volume_shape)
concat_merged_volume01 = merge_patches(concat_patches01, mapping, volume_shape)

temp_dir = "/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/concat_tmp_slices"
os.makedirs(temp_dir, exist_ok=True)

concat_1 = save_slices_as_images(concat_merged_volume1.cpu().squeeze().numpy())
concat_1[0].save("/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/concat_whole_gen11.gif", save_all=True, append_images=concat_1[1:], duration=200, loop=0)

concat_01 = save_slices_as_images(concat_merged_volume01.cpu().squeeze().numpy())
concat_01[0].save("/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/concat_whole_gen011.gif", save_all=True, append_images=concat_01[1:], duration=200, loop=0)

for img_file in os.listdir(temp_dir):
    os.remove(os.path.join(temp_dir, img_file))
os.rmdir(temp_dir)



# # noguide
# noguide_patches1 = []
# noguide_patches01 = []

# for idx in range(27):
#     # Generate or load the patch corresponding to index idx
#     # For example, use your model to generate the patch
#     noguide_patch1 = torch.load(f"/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/patch_generation/noguide_patch_{idx}_gen1_tensor.pth").squeeze(0)
#     noguide_patch01 = torch.load(f"/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/patch_generation/noguide_patch_{idx}_gen01_tensor.pth").squeeze(0)
#     noguide_patches1.append(noguide_patch1)
#     noguide_patches01.append(noguide_patch01)

# volume_shape = (1, 218, 182, 182)
# noguide_merged_volume1 = merge_patches(noguide_patches1, mapping, volume_shape)
# noguide_merged_volume01 = merge_patches(noguide_patches01, mapping, volume_shape)

# temp_dir = "/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/noguide_tmp_slices"
# os.makedirs(temp_dir, exist_ok=True)

# noguide_1 = save_slices_as_images(noguide_merged_volume1.cpu().squeeze().numpy())
# noguide_1[0].save("/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/noguide_whole_gen1.gif", save_all=True, append_images=noguide_1[1:], duration=200, loop=0)

# noguide_01 = save_slices_as_images(noguide_merged_volume01.cpu().squeeze().numpy())
# noguide_01[0].save("/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/noguide_whole_gen01.gif", save_all=True, append_images=noguide_01[1:], duration=200, loop=0)

# for img_file in os.listdir(temp_dir):
#     os.remove(os.path.join(temp_dir, img_file))
# os.rmdir(temp_dir)



# # encoder
# encoder_patches1 = []
# encoder_patches01 = []

# for idx in range(27):
#     # Generate or load the patch corresponding to index idx
#     # For example, use your model to generate the patch
#     encoder_patch1 = torch.load(f"/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/patch_generation/encoder_patch_{idx}_gen1_tensor.pth").squeeze(0)
#     encoder_patch01 = torch.load(f"/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/patch_generation/encoder_patch_{idx}_gen01_tensor.pth").squeeze(0)
#     encoder_patches1.append(encoder_patch1)
#     encoder_patches01.append(encoder_patch01)

# volume_shape = (1, 218, 182, 182)
# encoder_merged_volume1 = merge_patches(encoder_patches1, mapping, volume_shape)
# encoder_merged_volume01 = merge_patches(encoder_patches01, mapping, volume_shape)

# temp_dir = "/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/encoder_tmp_slices"
# os.makedirs(temp_dir, exist_ok=True)

# encoder_1 = save_slices_as_images(encoder_merged_volume1.cpu().squeeze().numpy())
# encoder_1[0].save("/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/encoder_whole_gen1.gif", save_all=True, append_images=encoder_1[1:], duration=200, loop=0)

# encoder_01 = save_slices_as_images(encoder_merged_volume01.cpu().squeeze().numpy())
# encoder_01[0].save("/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/encoder_whole_gen01.gif", save_all=True, append_images=encoder_01[1:], duration=200, loop=0)

# for img_file in os.listdir(temp_dir):
#     os.remove(os.path.join(temp_dir, img_file))
# os.rmdir(temp_dir)

