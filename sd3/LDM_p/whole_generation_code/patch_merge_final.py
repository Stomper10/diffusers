import os
from PIL import Image

import torch
import matplotlib.pyplot as plt

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

def collect_slice_means(patches, mapping, volume_depth):
    slice_means = [[] for _ in range(volume_depth)]  # Initialize list for each depth slice

    for idx, patch in enumerate(patches):
        start_d = mapping[idx]['start_d']
        D_p = patch.shape[1]
        for d in range(D_p):
            volume_slice_idx = start_d + d
            if volume_slice_idx < volume_depth:
                slice_mean = patch[:, d, :, :].mean().item()
                slice_means[volume_slice_idx].append(slice_mean)
    
    return slice_means

def compute_global_slice_means(slice_means):
    global_slice_means = []
    for means in slice_means:
        if means:
            global_mean = sum(means) / len(means)
        else:
            global_mean = 0.0  # Handle slices with no contributing patches
        global_slice_means.append(global_mean)
    
    return global_slice_means

def smooth_global_slice_means(global_slice_means, kernel_size=5, sigma=2.0):
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    smoothed_means = gaussian_filter1d(global_slice_means, sigma=sigma, truncate=kernel_size/(2*sigma))
    return smoothed_means.tolist()

def adjust_patches_global_slice_means(patches, mapping, global_slice_means):
    adjusted_patches = []
    for idx, patch in enumerate(patches):
        adjusted_patch = patch.clone()
        start_d = mapping[idx]['start_d']
        D_p = patch.shape[1]
        for d in range(D_p):
            volume_slice_idx = start_d + d
            if volume_slice_idx < len(global_slice_means):
                patch_slice = adjusted_patch[:, d, :, :]
                patch_slice_mean = patch_slice.mean()
                global_mean = global_slice_means[volume_slice_idx]

                # Compute scaling factor
                scaling_factor = global_mean / (patch_slice_mean + 1e-8)
                # Optionally limit the scaling factor
                scaling_factor = torch.clamp(scaling_factor, 0.9, 1.1)

                # Apply scaling factor
                adjusted_patch[:, d, :, :] *= scaling_factor.item()
        adjusted_patches.append(adjusted_patch)
    
    return adjusted_patches

def create_weight_mask(patch_size, device='cpu'):
    """
    Creates a 3D weight mask using a Hanning window.

    Args:
        patch_size (tuple): (D_p, H_p, W_p) of the patch.
        device (str): 'cpu' or 'cuda'.

    Returns:
        Tensor: Weight mask of shape (1, D_p, H_p, W_p)
    """
    D_p, H_p, W_p = patch_size

    # Generate 1D Hanning windows
    w_d = torch.hann_window(D_p, periodic=False, dtype=torch.float32).to(device)
    w_h = torch.hann_window(H_p, periodic=False, dtype=torch.float32).to(device)
    w_w = torch.hann_window(W_p, periodic=False, dtype=torch.float32).to(device)

    # Create 3D weight mask using outer product
    weight_mask = w_d[:, None, None] * w_h[None, :, None] * w_w[None, None, :]

    # Normalize the weight mask
    weight_mask = weight_mask / weight_mask.max()

    # Add channel dimension
    weight_mask = weight_mask.unsqueeze(0)  # Shape: (1, D_p, H_p, W_p)

    return weight_mask

def merge_patches_weighted(patches, mapping, volume_shape, patch_size, device='cpu'):
    """
    Merges a list of patches into a single volume using weighted blending.

    Args:
        patches (list of tensors): List of patch tensors with shape (C, D_p, H_p, W_p).
        mapping (list of dicts): List of mappings with 'start_d', 'start_h', 'start_w'.
        volume_shape (tuple): Shape of the whole volume (C, D, H, W).
        patch_size (tuple): Size of the patches (D_p, H_p, W_p).
        device (str): 'cpu' or 'cuda'.

    Returns:
        Tensor: Merged volume tensor with shape volume_shape.
    """
    channels, depth, height, width = volume_shape
    whole_volume = torch.zeros((channels, depth, height, width), dtype=patches[0].dtype, device=device)
    weight_volume = torch.zeros((channels, depth, height, width), dtype=patches[0].dtype, device=device)

    # Create weight mask once (assuming all patches are the same size)
    weight_mask = create_weight_mask(patch_size, device=device)

    for idx, patch in enumerate(patches):
        mapping_entry = mapping[idx]
        start_d = mapping_entry['start_d']
        start_h = mapping_entry['start_h']
        start_w = mapping_entry['start_w']

        pd, ph, pw = patch.shape[1:]  # Get patch dimensions

        # Apply weight mask to the patch
        weighted_patch = patch.to(device) * weight_mask

        # Add the weighted patch to the whole volume
        whole_volume[
            :, 
            start_d:start_d+pd, 
            start_h:start_h+ph, 
            start_w:start_w+pw
        ] += weighted_patch

        # Update the weight volume
        weight_volume[
            :, 
            start_d:start_d+pd, 
            start_h:start_h+ph, 
            start_w:start_w+pw
        ] += weight_mask

    # Avoid division by zero
    weight_volume[weight_volume == 0] = 1

    # Normalize to get the final merged volume
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



# Concat
import os
import numpy as np

original = os.listdir("/shared/s1/lab06/20252_individual_samples/")[:250]

dir_idx = os.listdir("/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/patch_generation/noguide_500/")
dir_idx_name = sorted(list(set([int(name.split("_")[2]) for name in dir_idx])))[:250]
len(dir_idx_name)

patch_size = (76, 64, 64)
volume_shape = (1, 218, 182, 182)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
volume_depth = volume_shape[1]  # Total depth of the volume

concat_numpy = []
for name in dir_idx_name:
    print(f"{name} processing")
    concat_patches = []
    for idx in range(27):
        # Generate or load the patch corresponding to index idx
        # For example, use your model to generate the patch
        concat_patch = torch.load(f"/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/patch_generation/noguide_500/noguide_patch_{name}_{idx}_gen1_tensor.pth").squeeze(0)
        concat_patches.append(concat_patch)
        
    # Collect slice means
    slice_means = collect_slice_means(concat_patches, mapping, volume_depth)

    # Compute global slice means
    global_slice_means = compute_global_slice_means(slice_means)

    # Optionally smooth the global slice means
    smoothed_global_slice_means = smooth_global_slice_means(global_slice_means, kernel_size=5, sigma=2.0)

    # Adjust patches using smoothed global slice means
    adjusted_patches = adjust_patches_global_slice_means(concat_patches, mapping, smoothed_global_slice_means)

    # Proceed to merge the adjusted patches as before
    merged_volume = merge_patches_weighted(adjusted_patches, mapping, volume_shape, patch_size, device=device)
    #print(merged_volume.shape)
    array = merged_volume.cpu().numpy()
    np.save(f'/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/noguide_numpy/noguide_{name}.npy', array)
    
    #concat_numpy.append(array)
    #concat_merged_volume = merge_patches(concat_patches, mapping, volume_shape)




# temp_dir = "/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/concat_tmp_slices"
# os.makedirs(temp_dir, exist_ok=True)

# concat = save_slices_as_images(merged_volume.cpu().squeeze().numpy()) # concat_merged_volume
# concat[0].save("/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/concat_whole_0_gen_temp_smooth.gif", save_all=True, append_images=concat[1:], duration=200, loop=0)

# for img_file in os.listdir(temp_dir):
#     os.remove(os.path.join(temp_dir, img_file))
# os.rmdir(temp_dir)




# # noguide
# noguide_patches = []

# for idx in range(27):
#     # Generate or load the patch corresponding to index idx
#     # For example, use your model to generate the patch
#     noguide_patch = torch.load(f"/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/patch_generation/noguide_500/noguide_patch_0_{idx}_gen1_tensor.pth").squeeze(0)
#     noguide_patches.append(noguide_patch)

# patch_size = (76, 64, 64)
# volume_shape = (1, 218, 182, 182)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# volume_depth = volume_shape[1]  # Total depth of the volume

# # Collect slice means
# slice_means = collect_slice_means(noguide_patches, mapping, volume_depth)

# # Compute global slice means
# global_slice_means = compute_global_slice_means(slice_means)

# # Optionally smooth the global slice means
# smoothed_global_slice_means = smooth_global_slice_means(global_slice_means, kernel_size=5, sigma=2.0)

# # Adjust patches using smoothed global slice means
# adjusted_patches = adjust_patches_global_slice_means(noguide_patches, mapping, smoothed_global_slice_means)

# # Proceed to merge the adjusted patches as before
# merged_volume = merge_patches_weighted(adjusted_patches, mapping, volume_shape, patch_size, device=device)

# noguide_merged_volume = merge_patches(noguide_patches, mapping, volume_shape)

# temp_dir = "/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/noguide_tmp_slices"
# os.makedirs(temp_dir, exist_ok=True)

# noguide = save_slices_as_images(merged_volume.cpu().squeeze().numpy())
# noguide[0].save("/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/noguide_whole_0_gen_temp_smooth.gif", save_all=True, append_images=noguide[1:], duration=200, loop=0)

# for img_file in os.listdir(temp_dir):
#     os.remove(os.path.join(temp_dir, img_file))
# os.rmdir(temp_dir)


# # encoder
# encoder_patches = []

# for idx in range(27):
#     # Generate or load the patch corresponding to index idx
#     # For example, use your model to generate the patch
#     encoder_patch = torch.load(f"/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/patch_generation/test/encoder_patch_{idx}_gen1_tensor.pth").squeeze(0)
#     encoder_patches.append(encoder_patch)

# patch_size = (76, 64, 64)
# volume_shape = (1, 218, 182, 182)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# volume_depth = volume_shape[1]  # Total depth of the volume

# # Collect slice means
# slice_means = collect_slice_means(encoder_patches, mapping, volume_depth)

# # Compute global slice means
# global_slice_means = compute_global_slice_means(slice_means)

# # Optionally smooth the global slice means
# smoothed_global_slice_means = smooth_global_slice_means(global_slice_means, kernel_size=5, sigma=2.0)

# # Adjust patches using smoothed global slice means
# adjusted_patches = adjust_patches_global_slice_means(encoder_patches, mapping, smoothed_global_slice_means)

# # Proceed to merge the adjusted patches as before
# merged_volume = merge_patches_weighted(adjusted_patches, mapping, volume_shape, patch_size, device=device)

# encoder_merged_volume = merge_patches(encoder_patches, mapping, volume_shape)

# temp_dir = "/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/encoder_tmp_slices"
# os.makedirs(temp_dir, exist_ok=True)

# encoder = save_slices_as_images(merged_volume.cpu().squeeze().numpy())
# encoder[0].save("/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/encoder_whole_0_gen_temp_smooth.gif", save_all=True, append_images=encoder[1:], duration=200, loop=0)

# for img_file in os.listdir(temp_dir):
#     os.remove(os.path.join(temp_dir, img_file))
# os.rmdir(temp_dir)
