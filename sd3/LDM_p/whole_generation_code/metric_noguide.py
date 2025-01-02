import os
import torch
import numpy as np
import torch.nn.functional as F
from monai import transforms
from skimage.metrics import structural_similarity as ssim

train_transforms = transforms.Compose(
        [
            transforms.ScaleIntensityd(keys=["pixel_values"], minv=-1.0, maxv=1.0),
            #transforms.Resized(keys=["pixel_values"], spatial_size=input_size, size_mode="all"),
            #transforms.CenterSpatialCropd(keys=["pixel_values"], roi_size=input_size),
            transforms.ThresholdIntensityd(keys=["pixel_values"], threshold=1, above=False, cval=1.0),
            transforms.ThresholdIntensityd(keys=["pixel_values"], threshold=-1, above=True, cval=-1.0),
            transforms.ToTensord(keys=["pixel_values"]),
        ]
    )

def volume_pairs_generator(original_dir, generated_dir):
    original_files = sorted(os.listdir(original_dir))[:100]
    generated_files = sorted(os.listdir(generated_dir))

    i = 0
    for orig_file, gen_file in zip(original_files, generated_files):
        print(gen_file, i)
        orig_path = os.path.join(original_dir, orig_file)
        gen_path = os.path.join(generated_dir, gen_file)

        original_volume = np.load(orig_path) # (128,128,128,1)
        image = torch.from_numpy(original_volume)
        axes_mapping = {
            's': (3, 0, 1, 2),
            'c': (3, 1, 0, 2),
            'a': (3, 2, 1, 0)
        }

        try:
            image = image.permute(*axes_mapping['c'])  # (1,128,128,128)
        except KeyError:
            raise ValueError("axis must be one of 'a', 'c', or 's'.")
        
        # Add batch dimension (N=1) for interpolation
        image = image.unsqueeze(0)  # Shape: (1, 1, D, H, W)

        # Define target size
        target_size = (218, 182, 182)  # (D₂, H₂, W₂)

        # Resize the volume using trilinear interpolation
        image = F.interpolate(
            image,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )  # Shape: (1, 1, D₂, H₂, W₂)
        sample = dict()
        sample["pixel_values"] = image.to(torch.float16)
        sample = train_transforms(sample)
        original_volume = np.array(sample["pixel_values"].squeeze(0))
        
        # generated 250 samples
        # generated_volume = torch.from_numpy(np.load(gen_path)) # (1,218,182,182)
        # gen = dict()
        # gen["pixel_values"] = generated_volume.unsqueeze(0).to(torch.float16)
        # gen = train_transforms(gen)
        # generated_volume = np.array(gen["pixel_values"].squeeze(0))
        generated_volume = np.load(gen_path) # (1,218,182,182)
        
        i += 1

        yield original_volume, generated_volume, orig_file  # Include filename for identification

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

        # Append metrics to lists
        mse_values.append(mse_value)
        rmse_values.append(rmse_value)
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        ncc_values.append(ncc_value)
        volume_names.append(volume_name)

    return {
        'volume_names': volume_names,
        'mse_values': mse_values,
        'rmse_values': rmse_values,
        'psnr_values': psnr_values,
        'ssim_values': ssim_values,
        'ncc_values': ncc_values
    }



# Run
original_dir = "/shared/s1/lab06/20252_individual_samples/"
generated_dir = "/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/whole_generation/E3_Gens_noguide"

mse_values = []
rmse_values = []
psnr_values = []
ssim_values = []
ncc_values = []
volume_names = []  # To keep track of volume identifiers

# Compute metrics
metrics_results = compute_metrics_for_volumes(original_dir, generated_dir)

# Access the results
volume_names = metrics_results['volume_names']
mse_values = metrics_results['mse_values']
rmse_values = metrics_results['rmse_values']
psnr_values = metrics_results['psnr_values']
ssim_values = metrics_results['ssim_values']
ncc_values = metrics_results['ncc_values']


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

print(f"Mean MSE: {mean_mse:.4f} ± {std_mse:.4f}")
print(f"Mean RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
print(f"Mean PSNR: {mean_psnr:.2f} dB ± {std_psnr:.2f} dB")
print(f"Mean SSIM: {mean_ssim:.4f} ± {std_ssim:.4f}")
print(f"Mean NCC: {mean_ncc:.4f} ± {std_ncc:.4f}")

# Find indices of volumes with highest MSE
sorted_indices = np.argsort(mse_values)[::-1]  # Descending order

print("Volumes with highest MSE:")
for idx in sorted_indices[:5]:  # Top 5
    print(f"{volume_names[idx]}: MSE = {mse_values[idx]:.4f}")
