import os
import torch
import matplotlib.pyplot as plt
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False



# Step 1: Create the mapping
mapping = []
index = 0

depth_starts = [0, 71, 142]
height_starts = [0, 59, 118]
width_starts = [0, 59, 118]

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

# Define the patch size
patch_size = (76, 64, 64)  # (D_p, H_p, W_p)

# Implement the functions as above (find_neighbors, get_overlap_coords, adjust_patch_brightness, adjust_patches_based_on_overlaps)

def find_neighbors(idx, mapping, patch_size):
    neighbors = []
    D_p, H_p, W_p = patch_size
    current_patch = mapping[idx]
    start_d1, start_h1, start_w1 = current_patch['start_d'], current_patch['start_h'], current_patch['start_w']
    end_d1 = start_d1 + D_p
    end_h1 = start_h1 + H_p
    end_w1 = start_w1 + W_p

    for i, other_patch in enumerate(mapping):
        if i == idx:
            continue  # Skip the current patch
        start_d2, start_h2, start_w2 = other_patch['start_d'], other_patch['start_h'], other_patch['start_w']
        end_d2 = start_d2 + D_p
        end_h2 = start_h2 + H_p
        end_w2 = start_w2 + W_p

        # Check for overlap in all three dimensions
        overlap_d = (start_d1 < end_d2) and (start_d2 < end_d1)
        overlap_h = (start_h1 < end_h2) and (start_h2 < end_h1)
        overlap_w = (start_w1 < end_w2) and (start_w2 < end_w1)

        if overlap_d and overlap_h and overlap_w:
            neighbors.append(i)

    return neighbors


def get_overlap_coords(mapping1, mapping2, patch_size):
    D_p, H_p, W_p = patch_size

    # Coordinates for the first patch
    start_d1, start_h1, start_w1 = mapping1['start_d'], mapping1['start_h'], mapping1['start_w']
    end_d1 = start_d1 + D_p
    end_h1 = start_h1 + H_p
    end_w1 = start_w1 + W_p

    # Coordinates for the second patch
    start_d2, start_h2, start_w2 = mapping2['start_d'], mapping2['start_h'], mapping2['start_w']
    end_d2 = start_d2 + D_p
    end_h2 = start_h2 + H_p
    end_w2 = start_w2 + W_p

    # Calculate the overlapping coordinates in the global volume
    overlap_start_d = max(start_d1, start_d2)
    overlap_end_d = min(end_d1, end_d2)
    overlap_start_h = max(start_h1, start_h2)
    overlap_end_h = min(end_h1, end_h2)
    overlap_start_w = max(start_w1, start_w2)
    overlap_end_w = min(end_w1, end_w2)

    # Check if there is no overlap
    if overlap_start_d >= overlap_end_d or overlap_start_h >= overlap_end_h or overlap_start_w >= overlap_end_w:
        return None  # No overlap

    # Convert global overlap coordinates to local patch coordinates
    patch1_coords = {
        'd_start': overlap_start_d - start_d1,
        'd_end': overlap_end_d - start_d1,
        'h_start': overlap_start_h - start_h1,
        'h_end': overlap_end_h - start_h1,
        'w_start': overlap_start_w - start_w1,
        'w_end': overlap_end_w - start_w1,
    }

    patch2_coords = {
        'd_start': overlap_start_d - start_d2,
        'd_end': overlap_end_d - start_d2,
        'h_start': overlap_start_h - start_h2,
        'h_end': overlap_end_h - start_h2,
        'w_start': overlap_start_w - start_w2,
        'w_end': overlap_end_w - start_w2,
    }

    return {'patch1': patch1_coords, 'patch2': patch2_coords}


def adjust_patch_brightness(patch, neighbor_patch, overlap_coords):
    # Extract overlapping regions
    coords1 = overlap_coords['patch1']
    coords2 = overlap_coords['patch2']

    # Get the overlapping regions in the patches
    patch_overlap = patch[
        :, 
        coords1['d_start']:coords1['d_end'],
        coords1['h_start']:coords1['h_end'],
        coords1['w_start']:coords1['w_end']
    ]
    neighbor_overlap = neighbor_patch[
        :, 
        coords2['d_start']:coords2['d_end'],
        coords2['h_start']:coords2['h_end'],
        coords2['w_start']:coords2['w_end']
    ]

    # Compute mean intensities
    mean_patch = patch_overlap.mean()
    mean_neighbor = neighbor_overlap.mean()

    # Compute scaling factor
    scaling_factor = mean_neighbor / (mean_patch + 1e-8)

    # Adjust the entire patch
    adjusted_patch = patch * scaling_factor

    return adjusted_patch


def adjust_patches_based_on_overlaps(patches, mapping, patch_size):
    adjusted_patches = patches.copy()
    for idx, patch in enumerate(patches):
        # Find neighboring patches
        neighbors = find_neighbors(idx, mapping, patch_size)
        for neighbor_idx in neighbors:
            neighbor_patch = patches[neighbor_idx]
            # Determine overlap coordinates
            overlap_coords = get_overlap_coords(mapping[idx], mapping[neighbor_idx], patch_size)
            if overlap_coords is None:
                continue  # No overlap
            # Adjust brightness
            adjusted_patch = adjust_patch_brightness(patch, neighbor_patch, overlap_coords)
            adjusted_patches[idx] = adjusted_patch
            # Optionally, adjust neighbor_patch as well
            # adjusted_neighbor_patch = adjust_patch_brightness(neighbor_patch, patch, overlap_coords)
            # adjusted_patches[neighbor_idx] = adjusted_neighbor_patch
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


##################################################################################

# Concat
concat_patches = []

for idx in range(27):
    concat_patch = torch.load(f"/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/patch_generation/concat_patch_{idx}_gen1_tensor.pth").squeeze(0)
    concat_patches.append(concat_patch) #volume_shape = (1, 218, 182, 182)
print("### torch.load done.")

# Define the patch size
patch_size = (76, 64, 64)  # (D_p, H_p, W_p)

# Adjust patches based on overlaps
adjusted_patches = adjust_patches_based_on_overlaps(concat_patches, mapping, patch_size)
print("### adjust_patches_based_on_overlaps done.")

# Define the volume shape
channels = concat_patches[0].shape[0]
volume_shape = (channels, 218, 182, 182)  # (C, D, H, W)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Merge the adjusted patches
merged_volume = merge_patches_weighted(concat_patches, mapping, volume_shape, patch_size, device=device)
print("### merge_patches_weighted done.")
# Now, merged_volume contains the final reconstructed volume with brightness adjustments and smooth boundaries

# Save as gif
temp_dir = "/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/concat_tmp_slices"
os.makedirs(temp_dir, exist_ok=True)

concat_1 = save_slices_as_images(merged_volume.cpu().squeeze().numpy())
concat_1[0].save("/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/concat_whole_gen1_smooth.gif", save_all=True, append_images=concat_1[1:], duration=200, loop=0)

for img_file in os.listdir(temp_dir):
    os.remove(os.path.join(temp_dir, img_file))
os.rmdir(temp_dir)

