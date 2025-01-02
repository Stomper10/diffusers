import os
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

def get_center_patch_slice_means(patches, volume_depth):
    """
    Compute slice-by-slice mean intensities from the center patch as the reference.
    
    Args:
        patches (list[Tensor]): List of patch tensors, each (C, D_p, H_p, W_p).
        volume_depth (int): Total depth of the final volume.
    
    Returns:
        list[float]: center_slice_means, a list of mean intensities per volume slice from the center patch.
                     If a slice does not exist in the center patch, its mean is set to 0.
    """
    center_patch_1 = patches[4]
    center_patch_2 = patches[13]
    center_patch_3 = patches[22]

    # Trim overlapping voxels (5 voxels) from the start of vol2 and vol3
    center_patch_2 = center_patch_2[:, 5:, :, :]  # Remove first 5 slices
    center_patch_3 = center_patch_3[:, 5:, :, :]  # Remove first 5 slices
    concatenated_volume = torch.cat([center_patch_1, center_patch_2, center_patch_3], dim=1)

    center_slice_means = [0.0] * volume_depth
    for d in range(volume_depth):
        center_slice_means[d] = concatenated_volume[:, d, :, :].mean().item()
    
    return center_slice_means

def adjust_patches_to_center_patch(patches, mapping, center_slice_means):
    """
    Adjust each patch so that each of its slices matches the center slice mean intensity
    defined by the center patch.

    Args:
        patches (list[Tensor]): List of patch tensors, each (C, D_p, H_p, W_p).
        mapping (list[dict]): Mapping info for each patch.
        center_slice_means (list[float]): Reference slice means from the center patch.

    Returns:
        list[Tensor]: Adjusted patches.
    """
    adjusted_patches = []
    for idx, patch in enumerate(patches):
        adjusted_patch = patch.clone()
        start_d = mapping[idx]['start_d']
        D_p = patch.shape[1]
        for d in range(D_p):
            volume_slice_idx = start_d + d
            if volume_slice_idx == 0:
                print("### volume_slice_idx:", volume_slice_idx)
            if volume_slice_idx < len(center_slice_means):
                ref_mean = center_slice_means[volume_slice_idx]
                if ref_mean == 0.0:
                    # If no reference mean (0), skip or just match mean to zero if desired
                    continue
                patch_slice = adjusted_patch[:, d, :, :]
                patch_slice_mean = patch_slice.mean()

                scaling_factor = ref_mean / (patch_slice_mean + 1e-8)
                # Optionally limit scaling factor to avoid extreme changes
                print("scaling_factor bc:", scaling_factor)
                scaling_factor = torch.clamp(torch.tensor(scaling_factor), 0.9, 1.1)
                print("scaling_factor ac:", scaling_factor)
                adjusted_patch[:, d, :, :] *= scaling_factor.item()
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
    center_slice_means = get_center_patch_slice_means(patches, volume_depth)

    # Adjust patches to match center patch's slice means
    adjusted_patches = adjust_patches_to_center_patch(patches, mapping, center_slice_means)

    # Merge adjusted patches
    merged_volume = merge_patches_weighted(adjusted_patches, mapping, volume_shape, patch_size, device=device)
    
    array = merged_volume.cpu().numpy()
    np.save(f'/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/E3_Gens_{mode}/E3_{mode}_whole_{name}_smooth.npy', array)
    break # test for single image

temp_dir = str(f"/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/{mode}_tmp_slices")
os.makedirs(temp_dir, exist_ok=True)

slices = save_slices_as_images(merged_volume.cpu().squeeze().numpy()) # {mode}_merged_volume
slices[0].save(f"/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/E3_{mode}_whole_{dir_idx_name[-1]}_smooth3.gif", 
               save_all=True, append_images=slices[1:], duration=200, loop=0)

for img_file in os.listdir(temp_dir):
    os.remove(os.path.join(temp_dir, img_file))
os.rmdir(temp_dir)
