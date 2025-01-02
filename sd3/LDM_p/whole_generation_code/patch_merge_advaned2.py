import os
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

def get_center_patch_slice_means(patches, mapping, volume_depth, center_patch_index=13):
    """
    Compute slice-by-slice mean intensities from the center patch as the reference.
    
    Args:
        patches (list[Tensor]): List of patch tensors, each (C, D_p, H_p, W_p).
        mapping (list[dict]): Mapping info for each patch (with 'start_d', etc.).
        volume_depth (int): Total depth of the final volume.
        center_patch_index (int): Index of the center patch. Default 13 for a 3x3x3 (27 patches) arrangement.
    
    Returns:
        list[float]: center_slice_means, a list of mean intensities per volume slice from the center patch.
                     If a slice does not exist in the center patch, its mean is set to 0.
    """
    center_patch = patches[center_patch_index]
    start_d = mapping[center_patch_index]['start_d']
    D_p = center_patch.shape[1]

    center_slice_means = [0.0] * volume_depth
    for d in range(D_p):
        volume_slice_idx = start_d + d
        if volume_slice_idx < volume_depth:
            slice_mean = center_patch[:, d, :, :].mean().item()
            center_slice_means[volume_slice_idx] = slice_mean
    
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
            if volume_slice_idx < len(center_slice_means):
                ref_mean = center_slice_means[volume_slice_idx]
                if ref_mean == 0.0:
                    # If no reference mean (0), skip or just match mean to zero if desired
                    continue
                patch_slice = adjusted_patch[:, d, :, :]
                patch_slice_mean = patch_slice.mean()

                scaling_factor = ref_mean / (patch_slice_mean + 1e-8)
                # Optionally limit scaling factor to avoid extreme changes
                #scaling_factor = torch.clamp(torch.tensor(scaling_factor), 0.9, 1.1)
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



# encoder
dir_idx = os.listdir("/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/patch_generation/E3_Gens_encoder/")
dir_idx_name = sorted(list(set([int(name.split("_")[3]) for name in dir_idx])))[:250]
len(dir_idx_name)

patch_size = (76, 64, 64)
volume_shape = (1, 218, 182, 182)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
volume_depth = volume_shape[1]  # Total depth of the volume

# Example usage assuming:
# - dir_idx_name: list of identifiers for volumes
# - patches, mapping, volume_shape, patch_size, volume_depth, device defined elsewhere
# - center_patch_index chosen based on how patches are arranged

center_patch_index = 13  # For example, if your patch arrangement's center is patch #13

numpy = []
for name in dir_idx_name:
    print(f"{name} processing")
    patches = []
    for idx in range(27):
        # Generate or load the patch corresponding to index idx
        # For example, use your model to generate the patch
        patch = torch.load(f"/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/patch_generation/E3_Gens_encoder/E3_encoder_patch_{name}_{idx}_gen1.pth").squeeze(0)
        patches.append(patch)
        
    # Compute reference slice means from center patch
    center_slice_means = get_center_patch_slice_means(patches, mapping, volume_depth, center_patch_index=center_patch_index)

    # Adjust patches to match center patch's slice means
    adjusted_patches = adjust_patches_to_center_patch(patches, mapping, center_slice_means)

    # Merge adjusted patches
    merged_volume = merge_patches_weighted(adjusted_patches, mapping, volume_shape, patch_size, device=device)
    
    array = merged_volume.cpu().numpy()
    np.save(f'/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/E3_Gens_encoder/E3_encoder_whole_{name}_smooth.npy', array)
    break # test for single image

temp_dir = "/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/encoder_tmp_slices"
os.makedirs(temp_dir, exist_ok=True)

encoder = save_slices_as_images(merged_volume.cpu().squeeze().numpy()) # encoder_merged_volume
encoder[0].save(f"/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/E3_encoder_whole_{dir_idx_name[-1]}_smooth2.gif", 
                save_all=True, append_images=encoder[1:], duration=200, loop=0)

for img_file in os.listdir(temp_dir):
    os.remove(os.path.join(temp_dir, img_file))
os.rmdir(temp_dir)



# # noguide
# dir_idx = os.listdir("/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/patch_generation/E3_Gens_noguide/")
# dir_idx_name = sorted(list(set([int(name.split("_")[2]) for name in dir_idx])))[:250]
# len(dir_idx_name)

# patch_size = (76, 64, 64)
# volume_shape = (1, 218, 182, 182)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# volume_depth = volume_shape[1]  # Total depth of the volume

# numpy = []
# for name in dir_idx_name:
#     print(f"{name} processing")
#     patches = []
#     for idx in range(27):
#         # Generate or load the patch corresponding to index idx
#         # For example, use your model to generate the patch
#         patch = torch.load(f"/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/patch_generation/E3_Gens_noguide/E3_noguide_patch_{name}_{idx}_gen1.pth").squeeze(0)
#         patches.append(patch)
        
#     # Collect slice means
#     slice_means = collect_slice_means(patches, mapping, volume_depth)

#     # Compute global slice means
#     global_slice_means = compute_global_slice_means(slice_means)

#     # Optionally smooth the global slice means
#     smoothed_global_slice_means = smooth_global_slice_means(global_slice_means, kernel_size=5, sigma=2.0)

#     # Adjust patches using smoothed global slice means
#     adjusted_patches = adjust_patches_global_slice_means(patches, mapping, smoothed_global_slice_means)

#     # Proceed to merge the adjusted patches as before
#     merged_volume = merge_patches_weighted(adjusted_patches, mapping, volume_shape, patch_size, device=device)
#     #merged_volume = merge_patches(patches, mapping, volume_shape)
#     #print(merged_volume.shape)
#     array = merged_volume.cpu().numpy()
#     np.save(f'/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/E3_Gens_noguide/E3_noguide_whole_{name}_smooth.npy', array)

# temp_dir = "/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/noguide_tmp_slices"
# os.makedirs(temp_dir, exist_ok=True)

# noguide = save_slices_as_images(merged_volume.cpu().squeeze().numpy()) # noguide_merged_volume
# noguide[0].save(f"/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/E3_noguide_whole_{dir_idx_name[-1]}_smooth.gif", 
#                 save_all=True, append_images=noguide[1:], duration=200, loop=0)

# for img_file in os.listdir(temp_dir):
#     os.remove(os.path.join(temp_dir, img_file))
# os.rmdir(temp_dir)
