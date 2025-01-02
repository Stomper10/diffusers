import os
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

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

###
mode = 'encoder'
ex_num = 8
#mode = 'noguide'

# merge
dir_idx = os.listdir(f"/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/patch_generation/E{ex_num}_Gens_{mode}/")
if mode == 'noguide':
    dir_idx_name = sorted(list(set([int(name.split("_")[3]) for name in dir_idx])))
    dir_idx_name.remove(804)
    dir_idx_name.remove(904)
    dir_idx_name = dir_idx_name[:100]
else:
    dir_idx_name = sorted(list(set([int(name.split("_")[3]) for name in dir_idx])))[:100]
print(len(dir_idx_name))
dir_idx_name = [0]

patch_size = (76, 64, 64)
volume_shape = (1, 218, 182, 182)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
volume_depth = volume_shape[1]  # Total depth of the volume

for name in dir_idx_name:
    print(f"{name} processing")
    patches = []
    for idx in range(27):
        # Generate or load the patch corresponding to index idx
        # For example, use your model to generate the patch
        patch = torch.load(f"/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/patch_generation/E{ex_num}_Gens_{mode}/E6_{mode}_patch_{name}_{idx}_gen1.pth").squeeze(0)
        patches.append(patch)

    # Merge adjusted patches
    merged_volume = merge_patches_weighted(patches, mapping, volume_shape, patch_size, device=device)
    
    array = merged_volume.cpu().numpy()
    np.save(f'/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/E{ex_num}_Gens_{mode}/E{ex_num}_{mode}_whole_{name}_smooth.npy', array)
    #break # test for single image

temp_dir = str(f"/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/{mode}_tmp_slices")
os.makedirs(temp_dir, exist_ok=True)



# GIF gen
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


slices = save_slices_as_images(merged_volume.cpu().squeeze().numpy()) # {mode}_merged_volume
slices[0].save(f"/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/E{ex_num}_{mode}_whole_{dir_idx_name[-1]}_smooth_naive.gif", 
               save_all=True, append_images=slices[1:], duration=200, loop=0)

for img_file in os.listdir(temp_dir):
    os.remove(os.path.join(temp_dir, img_file))
os.rmdir(temp_dir)
