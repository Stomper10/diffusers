import os
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import torch
import torch.nn as nn
import numpy as np

def get_center_patch_slice_means_and_stds(patches, volume_depth, threshold=0.05):
    """
    Compute slice-by-slice mean and std intensities from the combined center patches as the reference,
    considering only voxels above a given threshold to exclude background.

    Args:
        patches (list[Tensor]): List of patch tensors, each (C, D_p, H_p, W_p).
                                This code assumes the three center patches have already been chosen
                                (e.g., patches[4], patches[13], patches[22]) and you remove
                                slices from patch_2 and patch_3 as needed.
        volume_depth (int): Total depth of the final volume.
        threshold (float): Threshold to distinguish background from brain tissue.

    Returns:
        (list[float], list[float]): 
            center_slice_means and center_slice_stds: each is a list of per-slice stats.
            If no valid voxel in a slice, that slice mean and std are 0.
    """
    # Example: patches[4], patches[13], patches[22] are center patches
    center_patch_1 = patches[4]
    center_patch_2 = patches[13]
    center_patch_3 = patches[22]

    # Remove first 5 slices if required
    center_patch_2 = center_patch_2[:, 5:, :, :]
    center_patch_3 = center_patch_3[:, 5:, :, :]

    concatenated_volume = torch.cat([center_patch_1, center_patch_2, center_patch_3], dim=1)
    # concatenated_volume shape: (C, D_total, H, W)

    C, D_total, H, W = concatenated_volume.shape
    center_slice_means = [0.0] * volume_depth
    center_slice_stds = [0.0] * volume_depth

    for d in range(volume_depth):
        if d < D_total:
            slice_ = concatenated_volume[:, d, :, :]  # (C, H, W)
            # Create a mask for non-background voxels
            # Assuming single-channel data for simplicity:
            mask = (slice_[0] > threshold)

            valid_voxels = slice_[0, mask]
            if valid_voxels.numel() > 0:
                mean_val = valid_voxels.mean().item()
                std_val = valid_voxels.std().item()
                center_slice_means[d] = mean_val
                center_slice_stds[d] = std_val
            else:
                # No valid voxel in this slice
                center_slice_means[d] = 0.0
                center_slice_stds[d] = 0.0
        else:
            # If this slice index exceeds D_total, keep 0.0 for both mean and std
            center_slice_means[d] = 0.0
            center_slice_stds[d] = 0.0

    return center_slice_means, center_slice_stds



def adjust_patches_to_center_patch(patches, mapping, center_slice_means, center_slice_stds, threshold=0.05):
    """
    Adjust each patch so that each of its slices matches both the mean and std
    of the center slice intensity, considering only valid brain voxels above a threshold.

    Args:
        patches (list[Tensor]): Each patch is (C, D_p, H_p, W_p)
        mapping (list[dict]): Patch mapping with 'start_d', 'start_h', 'start_w'.
        center_slice_means (list[float]): Reference slice means from the center patch.
        center_slice_stds (list[float]): Reference slice standard deviations from the center patch.
        threshold (float): Threshold to distinguish background from brain tissue.

    Returns:
        list[Tensor]: Adjusted patches after mean and variance matching.
    """
    adjusted_patches = []
    for idx, patch in enumerate(patches):
        adjusted_patch = patch.clone()
        start_d = mapping[idx]['start_d']
        D_p = patch.shape[1]

        for d in range(D_p):
            volume_slice_idx = start_d + d
            if volume_slice_idx < len(center_slice_means):
                ref_mean = center_slice_means[volume_slice_idx]
                ref_std = center_slice_stds[volume_slice_idx]

                if ref_mean == 0.0 and ref_std < 1e-8:
                    # Reference slice is essentially uniform or zero, skip
                    continue

                patch_slice = adjusted_patch[:, d, :, :]
                
                # Create a mask for valid brain voxels
                # Assuming single-channel for background check. If multi-channel, adapt accordingly.
                mask = (patch_slice[0] > threshold)

                # If no valid voxels, skip adjustments
                if mask.sum().item() == 0:
                    continue

                valid_voxels = patch_slice[0, mask]
                patch_slice_mean = valid_voxels.mean()
                patch_slice_std = valid_voxels.std()

                # If patch slice is almost uniform, just match mean
                if patch_slice_std < 1e-8:
                    # Scale not defined, just shift mean
                    shift = ref_mean - patch_slice_mean
                    patch_slice[0, mask] = patch_slice[0, mask] + shift.item()
                else:
                    # Mean and variance matching
                    scale = ref_std / (patch_slice_std + 1e-8)
                    # Optionally clamp the scale if you want to avoid extreme changes
                    scale = torch.clamp(torch.tensor(scale), 0.5, 2.0).item()

                    # Apply mean + variance matching:
                    # (x - slice_mean)*scale + ref_mean
                    patch_slice[0, mask] = ((patch_slice[0, mask] - patch_slice_mean) * scale) + ref_mean

        adjusted_patches.append(adjusted_patch)
    
    return adjusted_patches



def create_weight_mask(patch_size, device='cpu'):
    """
    Creates a 3D weight mask using a Hanning window for blending patches.
    """
    D_p, H_p, W_p = patch_size
    w_d = torch.hann_window(D_p, periodic=False, dtype=torch.float32).to(device)
    w_h = torch.hann_window(H_p, periodic=False, dtype=torch.float32).to(device)
    w_w = torch.hann_window(W_p, periodic=False, dtype=torch.float32).to(device)

    weight_mask = w_d[:, None, None] * w_h[None, :, None] * w_w[None, None, :]
    weight_mask = weight_mask / weight_mask.max()
    weight_mask = weight_mask.unsqueeze(0)  # (1, D_p, H_p, W_p)
    return weight_mask

def merge_patches_weighted(patches, mapping, volume_shape, patch_size, device='cpu'):
    """
    Merge patches using weighted blending.
    """
    channels, depth, height, width = volume_shape
    dtype = patches[0].dtype

    whole_volume = torch.zeros((channels, depth, height, width), dtype=dtype, device=device)
    weight_volume = torch.zeros((channels, depth, height, width), dtype=dtype, device=device)

    weight_mask = create_weight_mask(patch_size, device=device)

    for idx, patch in enumerate(patches):
        mapping_entry = mapping[idx]
        start_d = mapping_entry['start_d']
        start_h = mapping_entry['start_h']
        start_w = mapping_entry['start_w']

        pd, ph, pw = patch.shape[1:]  # (D_p, H_p, W_p)
        weighted_patch = patch.to(device) * weight_mask

        whole_volume[:,
                     start_d:start_d+pd,
                     start_h:start_h+ph,
                     start_w:start_w+pw] += weighted_patch

        weight_volume[:,
                      start_d:start_d+pd,
                      start_h:start_h+ph,
                      start_w:start_w+pw] += weight_mask

    weight_volume[weight_volume == 0] = 1
    merged_volume = whole_volume / weight_volume
    return merged_volume

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



# mapping
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



mode = 'encoder'
#mode = 'noguide'

# merge
dir_idx = os.listdir(f"/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/patch_generation/E3_Gens_{mode}/")
dir_idx_name = sorted(list(set([int(name.split("_")[3]) for name in dir_idx])))[:250]
len(dir_idx_name)

patch_size = (76, 64, 64)
volume_shape = (1, 218, 182, 182)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
volume_depth = volume_shape[1]  # Total depth of the volume

# Example usage assuming:
# - dir_idx_name: list of identifiers for volumes
# - patches, mapping, volume_shape, patch_size, volume_depth, device defined elsewhere

for name in dir_idx_name:
    print(f"{name} processing")
    patches = []
    for idx in range(27):
        # Generate or load the patch corresponding to index idx
        # For example, use your model to generate the patch
        patch = torch.load(f"/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/patch_generation/E3_Gens_{mode}/E3_{mode}_patch_{name}_{idx}_gen1.pth").squeeze(0)
        patches.append(patch)
        
    # Compute reference slice means from center patch
    center_slice_means, center_slice_stds = get_center_patch_slice_means_and_stds(patches, volume_depth)

    # Adjust patches to match center patch's slice means
    adjusted_patches = adjust_patches_to_center_patch(patches, mapping, center_slice_means, center_slice_stds)

    # Merge adjusted patches
    merged_volume = merge_patches_weighted(adjusted_patches, mapping, volume_shape, patch_size, device=device)
    
    array = merged_volume.cpu().numpy()
    np.save(f'/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/E3_Gens_{mode}/E3_{mode}_whole_{name}_smooth.npy', array)
    break # test for single image

temp_dir = str(f"/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/{mode}_tmp_slices")
os.makedirs(temp_dir, exist_ok=True)

slices = save_slices_as_images(merged_volume.cpu().squeeze().numpy()) # {mode}_merged_volume
slices[0].save(f"/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/E3_{mode}_whole_{dir_idx_name[-1]}_smooth2.gif", 
               save_all=True, append_images=slices[1:], duration=200, loop=0)

for img_file in os.listdir(temp_dir):
    os.remove(os.path.join(temp_dir, img_file))
os.rmdir(temp_dir)
