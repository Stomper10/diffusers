import os
import torch
import numpy as np
import pandas as pd
import nibabel as nib
from monai import transforms
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.ssim import MultiScaleStructuralSimilarityIndexMeasure
from torch import rand

input_size = (128,128,128)
train_transforms = transforms.Compose(
        [
            transforms.Resized(keys=["pixel_values"], spatial_size=input_size, size_mode="all"),
            transforms.ScaleIntensityd(keys=["pixel_values"], minv=-1.0, maxv=1.0),
            # transforms.ThresholdIntensityd(keys=["pixel_values"], threshold=1, above=False, cval=1.0),
            # transforms.ThresholdIntensityd(keys=["pixel_values"], threshold=-1, above=True, cval=-1.0),
            transforms.ToTensord(keys=["pixel_values"]),
        ]
    )

def volume_pairs_generator(original_dir, generated_dir):
    original_files = original_dir[:100]
    generated_files = generated_dir[100:200]

    i = 0
    for orig_file, gen_file in zip(original_files, generated_files):
        print(i, orig_file.split("/")[4], gen_file.split("/")[4])
        
        axes_mapping = {
            's': (0, 1, 2),
            'c': (1, 0, 2),
            'a': (2, 1, 0)
        }

        # original
        image = nib.load(orig_file)
        image = image.get_fdata()
        image = torch.from_numpy(image) # Stays on CPU # (182, 218, 182)

        image = image.permute(*axes_mapping['c']).unsqueeze(0)  # (1, 182, 218, 182)
        sample = {"pixel_values": image.to(torch.float16),}
        sample = train_transforms(sample) # SET
        original_volume = np.array(sample["pixel_values"].squeeze(0))

        # generation
        image = nib.load(gen_file)
        image = image.get_fdata()
        image = torch.from_numpy(image) # Stays on CPU # (182, 218, 182)

        image = image.permute(*axes_mapping['c']).unsqueeze(0)  # (1, 182, 218, 182)
        sample = {"pixel_values": image.to(torch.float16),}
        sample = train_transforms(sample) # SET
        generated_volume = np.array(sample["pixel_values"].squeeze(0))

        if i == 0:
            slice_2d = original_volume[64]
            slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min()) * 255
            slice_2d = slice_2d.astype(np.uint8)
            image = Image.fromarray(slice_2d)
            image.save("/shared/s1/lab06/wonyoung/diffusers/sd3/LDM/gen_volumes/gif/original_1_slice.png")  # Save as PNG

            slice_2d = generated_volume[64]
            slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min()) * 255
            slice_2d = slice_2d.astype(np.uint8)
            image = Image.fromarray(slice_2d)
            image.save("/shared/s1/lab06/wonyoung/diffusers/sd3/LDM/gen_volumes/gif/original_2_slice.png")  # Save as PNG

        i += 1

        yield original_volume, generated_volume, gen_file  # Include filename for identification

def compute_mse(original, generated):
    """
    Computes the Mean Squared Error between two volumes.
    
    Args:
        original (numpy.ndarray): Original volume.
        generated (numpy.ndarray): Generated volume.
    
    Returns:
        float: MSE value.
    """
    mse_value = np.mean((original - generated) ** 2)
    return mse_value

def compute_rmse(original, generated):
    mse_value = compute_mse(original, generated)
    rmse_value = np.sqrt(mse_value)
    return rmse_value

def compute_psnr(original, generated, max_intensity=None):
    """
    Computes the Peak Signal-to-Noise Ratio between two volumes.
    
    Args:
        original (numpy.ndarray): Original volume.
        generated (numpy.ndarray): Generated volume.
        max_intensity (float): Maximum possible intensity value. If None, it's estimated from the data.
    
    Returns:
        float: PSNR value in decibels (dB).
    """
    mse_value = compute_mse(original, generated)
    if mse_value == 0:
        return float('inf')  # Perfect match
    if max_intensity is None:
        max_intensity = max(original.max(), generated.max())
    psnr_value = 20 * np.log10(max_intensity / np.sqrt(mse_value))
    return psnr_value

def compute_ssim(original, generated):
    """
    Computes the Structural Similarity Index Measure between two volumes.
    
    Args:
        original (numpy.ndarray): Original volume.
        generated (numpy.ndarray): Generated volume.
    
    Returns:
        float: SSIM value.
    """
    # Ensure the data is in the correct format
    original = original.squeeze().astype(np.float32)
    generated = generated.squeeze().astype(np.float32)
    
    ssim_value, _ = ssim(original, generated, data_range=generated.max() - generated.min(), full=True)
    return ssim_value

def compute_msssim(original, generated):
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=generated.max() - generated.min(),
                                                         kernel_size=7)
    ms_score = ms_ssim(torch.from_numpy(original).unsqueeze(0).unsqueeze(0), 
                       torch.from_numpy(generated).unsqueeze(0).unsqueeze(0))
    return  ms_score.item()

def compute_ncc(original, generated):
    """
    Computes the Normalized Cross-Correlation between two volumes.
    
    Args:
        original (numpy.ndarray): Original volume.
        generated (numpy.ndarray): Generated volume.
    
    Returns:
        float: NCC value.
    """
    original_flat = original.flatten()
    generated_flat = generated.flatten()
    
    mean_original = np.mean(original_flat)
    mean_generated = np.mean(generated_flat)
    
    numerator = np.sum((original_flat - mean_original) * (generated_flat - mean_generated))
    denominator = np.sqrt(np.sum((original_flat - mean_original) ** 2) * np.sum((generated_flat - mean_generated) ** 2))
    
    if denominator == 0:
        return 0.0
    ncc_value = numerator / denominator
    return ncc_value



# combined
def compute_metrics_for_volumes(original_dir, generated_dir):
    for original_volume, generated_volume, volume_name in volume_pairs_generator(original_dir, generated_dir):
        # Ensure volumes have the same shape
        assert original_volume.shape == generated_volume.shape, f"Volumes must have the same shape. Issue with {volume_name}"

        # Convert volumes to float32 if necessary
        original_volume = original_volume.astype(np.float32)
        generated_volume = generated_volume.astype(np.float32)

        # Compute metrics
        mse_value = compute_mse(original_volume, generated_volume)
        rmse_value = compute_rmse(original_volume, generated_volume)
        psnr_value = compute_psnr(original_volume, generated_volume)
        ssim_value = compute_ssim(original_volume, generated_volume)
        ncc_value = compute_ncc(original_volume, generated_volume)
        msssim_value = compute_msssim(original_volume, generated_volume)

        # Append metrics to lists
        mse_values.append(mse_value)
        rmse_values.append(rmse_value)
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        ncc_values.append(ncc_value)
        msssim_values.append(msssim_value)
        volume_names.append(volume_name)

    return {
        'volume_names': volume_names,
        'mse_values': mse_values,
        'rmse_values': rmse_values,
        'psnr_values': psnr_values,
        'ssim_values': ssim_values,
        'ncc_values': ncc_values,
        'msssim_values': msssim_values
    }



# Run
original_dir = "/leelabsg/data/20252_unzip"
generated_dir = "/leelabsg/data/20252_unzip"
label_dir = "/shared/s1/lab06/wonyoung/diffusers/sd3/data/train.csv"
data_csv = pd.read_csv(label_dir)
image_names = list(data_csv['rel_path'])
origin_paths = [os.path.join(original_dir, name) for name in image_names]
generated_paths = [os.path.join(generated_dir, name) for name in image_names]

mse_values = []
rmse_values = []
psnr_values = []
ssim_values = []
ncc_values = []
msssim_values = []
volume_names = []  # To keep track of volume identifiers

# Compute metrics
metrics_results = compute_metrics_for_volumes(origin_paths, generated_paths)

# Access the results
volume_names = metrics_results['volume_names']
mse_values = metrics_results['mse_values']
rmse_values = metrics_results['rmse_values']
psnr_values = metrics_results['psnr_values']
ssim_values = metrics_results['ssim_values']
ncc_values = metrics_results['ncc_values']
msssim_values = metrics_results['msssim_values']

mean_mse = np.mean(mse_values)
std_mse = np.std(mse_values)

mean_rmse = np.mean(rmse_values)
std_rmse = np.std(rmse_values)

mean_psnr = np.mean(psnr_values)
std_psnr = np.std(psnr_values)

mean_ssim = np.mean(ssim_values)
std_ssim = np.std(ssim_values)

mean_ncc = np.mean(ncc_values)
std_ncc = np.std(ncc_values)

mean_msssim = np.mean(msssim_values)
std_msssim = np.std(msssim_values)

print(f"Mean MSE: {mean_mse:.4f} ± {std_mse:.4f}")
print(f"Mean RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
print(f"Mean PSNR: {mean_psnr:.2f} dB ± {std_psnr:.2f} dB")
print(f"Mean SSIM: {mean_ssim:.4f} ± {std_ssim:.4f}")
print(f"Mean NCC: {mean_ncc:.4f} ± {std_ncc:.4f}")
print(f"Mean MS-SSIM: {mean_msssim:.4f} ± {std_msssim:.4f}")

# Find indices of volumes with highest MSE
sorted_indices = np.argsort(mse_values)[::-1]  # Descending order

print("Volumes with highest MSE:")
for idx in sorted_indices[:5]:  # Top 5
    print(f"{volume_names[idx]}: MSE = {mse_values[idx]:.4f}")



# slice-wise FID
def volume_pairs(original_dir, generated_dir):
    original_files = original_dir[:100]
    generated_files = generated_dir[100:200]

    i = 0
    origin_list, gen_list = [], []
    for orig_file, gen_file in zip(original_files, generated_files):
        print(i, orig_file.split("/")[4], gen_file.split("/")[4])
        
        axes_mapping = {
            's': (0, 1, 2),
            'c': (1, 0, 2),
            'a': (2, 1, 0)
        }

        # original
        image = nib.load(orig_file)
        image = image.get_fdata()
        image = torch.from_numpy(image) # Stays on CPU # (182, 218, 182)

        image = image.permute(*axes_mapping['c']).unsqueeze(0)  # (1, 182, 218, 182)
        sample = {"pixel_values": image.to(torch.float16),}
        sample = train_transforms(sample) # SET
        #original_volume = np.array(sample["pixel_values"].squeeze(0))
        origin_list.append(sample["pixel_values"])

        # generation
        image = nib.load(gen_file)
        image = image.get_fdata()
        image = torch.from_numpy(image) # Stays on CPU # (182, 218, 182)

        image = image.permute(*axes_mapping['c']).unsqueeze(0)  # (1, 182, 218, 182)
        sample = {"pixel_values": image.to(torch.float16),}
        sample = train_transforms(sample) # SET
        #generated_volume = np.array(sample["pixel_values"].squeeze(0))
        gen_list.append(sample["pixel_values"])

        i += 1

    return origin_list, gen_list

origin_list, gen_list = volume_pairs(origin_paths, generated_paths) # (100,1,128,128,128) tensor
org_tensor = torch.stack(origin_list, dim=0)  # [100, 1, 128, 128, 128]
org_tensor = ((org_tensor + 1) / 2 * 255).to(torch.uint8)
gen_tensor = torch.stack(gen_list, dim=0)  # [100, 1, 128, 128, 128]
gen_tensor = ((gen_tensor + 1) / 2 * 255).to(torch.uint8)
idx = 64

org_slice_2 = org_tensor[:, :, idx, :, :].repeat(1, 3, 1, 1)  # Shape: (100, 3, 128, 128)
gen_slice_2 = gen_tensor[:, :, idx, :, :].repeat(1, 3, 1, 1)  # Shape: (100, 3, 128, 128)

org_slice_3 = org_tensor[:, :, :, idx, :].repeat(1, 3, 1, 1)  # Shape: (100, 3, 128, 128)
gen_slice_3 = gen_tensor[:, :, :, idx, :].repeat(1, 3, 1, 1)  # Shape: (100, 3, 128, 128)

org_slice_4 = org_tensor[:, :, :, :, idx].repeat(1, 3, 1, 1)  # Shape: (100, 3, 128, 128)
gen_slice_4 = gen_tensor[:, :, :, :, idx].repeat(1, 3, 1, 1)  # Shape: (100, 3, 128, 128)

fid = FrechetInceptionDistance(feature=64)
fid.update(org_slice_2, real=True)
fid.update(gen_slice_2, real=False)
print(fid.compute()) # feature=2048 metatensor(24.8120) / feature=64 metatensor(0.0045)

fid = FrechetInceptionDistance(feature=64)
fid.update(org_slice_3, real=True)
fid.update(gen_slice_3, real=False)
print(fid.compute()) # feature=2048 metatensor(13.1972) / feature=64 metatensor(0.0037)

fid = FrechetInceptionDistance(feature=64)
fid.update(org_slice_4, real=True)
fid.update(gen_slice_4, real=False)
print(fid.compute()) # feature=2048 metatensor(18.4475) / feature=64 metatensor(0.0058)
