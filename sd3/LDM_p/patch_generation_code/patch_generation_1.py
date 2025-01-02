import argparse
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers.models.vq_gan_3d import VQGAN
from diffusers.models.ddpm import PatchUnet3D, PatchGaussianDiffusion
from monai import transforms

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--number", "-n", type=int, required=True, default=0, help="Generation number set.")
args = parser.parse_args()

pretrained_vae_path="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/E1_pLDM_VQGAN3D/checkpoint-770000"
pretrained_unet_path="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/E8_pLDM_UNET3D_whole/checkpoint-60000"
output_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/patch_generation/E8_Gens_encoder"
mixed_precision="fp16"
resolution="76,64,64"
num_timesteps=1000
input_size = tuple(int(x) for x in resolution.split(","))
accelerator = Accelerator(mixed_precision=mixed_precision,)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae = VQGAN.from_pretrained(pretrained_vae_path, subfolder="vqgan",).to(device)
unet3d = PatchUnet3D.from_pretrained(pretrained_unet_path, subfolder="unet3d",).to(device)
vae.eval()
unet3d.eval()
noise_scheduler = PatchGaussianDiffusion(timesteps=1000).to(device)

unet3d, vae = accelerator.prepare(unet3d, vae)
vae_model = accelerator.unwrap_model(vae)
unet3d_model = accelerator.unwrap_model(unet3d)

# Generation
# Initialize lists
depth_starts = [0, 71, 142]
height_starts = [0, 59, 118]
width_starts = [0, 59, 118]

# Initialize index
index = 0
# Mapping list to store results
mapping = []

for id, start_d in enumerate(depth_starts):
    for ih, start_h in enumerate(height_starts):
        for iw, start_w in enumerate(width_starts):
            mapping.append({
                'index': index,
                'start_d': start_d,
                'start_h': start_h,
                'start_w': start_w
            })
            print(f"Index: {index}, Start Coordinates: (D: {start_d}, H: {start_h}, W: {start_w})")
            index += 1

def get_patch_by_index(volume, index, mapping):
    # Retrieve the start coordinates from the mapping
    mapping_entry = mapping[index]
    start_d = mapping_entry['start_d']
    start_h = mapping_entry['start_h']
    start_w = mapping_entry['start_w']
    
    # Patch dimensions
    pd, ph, pw = 76, 64, 64
    
    # Extract the patch
    patch = volume[
        :, 
        start_d:start_d + pd,
        start_h:start_h + ph,
        start_w:start_w + pw
    ]
    return patch


print("Inference starts")
#gen_images = []
num_samples = 100

guide_load = torch.load("/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_w/results/low-res_gen/tensor_low-res_image_1_scale2.pth").to(torch.float64).to(unet3d.device) # (1,1,76,64,64)
# guide_inter = F.interpolate( # (1,1,218,182,182)
#     guide_load,
#     size=(218, 182, 182),
#     mode='trilinear',
#     align_corners=False
# )
guide_raw = guide_load.squeeze(0).to(torch.float16) # (1,76,64,64)

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

guide_dict = dict()
guide_dict["pixel_values"] = guide_raw
guide_dict = train_transforms(guide_dict)

def gaussian_kernel_3d(kernel_size=5, sigma=2.0, device='cpu'):
    """
    Create a 3D Gaussian kernel for convolution.
    kernel_size: int (odd number), size of the kernel for D,H,W.
    sigma: float, standard deviation for Gaussian.
    """
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device)
    coords = coords - (kernel_size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    
    # g is 1D, make it 3D by outer products
    g_3d = g[:, None, None] * g[None, :, None] * g[None, None, :]
    g_3d = g_3d / g_3d.sum()
    
    # Shape (1,1,D,H,W)
    g_3d = g_3d.unsqueeze(0).unsqueeze(0)
    return g_3d

def gaussian_blur_3d(volume, kernel_size=5, sigma=2.0):
    """
    Apply 3D Gaussian blur to a volume of shape (1, D, H, W).
    
    Args:
        volume (torch.Tensor): shape (1, D, H, W), single-channel volume.
        kernel_size (int): Kernel size for Gaussian.
        sigma (float): Standard deviation for Gaussian.
        
    Returns:
        torch.Tensor: Blurred volume of shape (1, D, H, W).
    """
    device = volume.device
    C = volume.shape[0]
    assert C == 1, "This function assumes a single channel (C=1)."
    assert volume.ndim == 4, "Volume should have shape (1, D, H, W)."
    
    kernel = gaussian_kernel_3d(kernel_size, sigma, device=device)  # (1,1,D,H,W)
    
    # Add a batch dimension: (1, C=1, D, H, W)
    volume = volume.unsqueeze(0)  # shape now (1,1,D,H,W)
    
    # Perform convolution with padding to maintain shape
    padding = kernel_size // 2
    blurred = F.conv3d(volume, kernel, padding=padding, groups=1)
    # blurred shape is still (1,1,D,H,W)
    
    # Remove batch dimension
    blurred = blurred.squeeze(0)  # back to (1, D, H, W)
    return blurred

with torch.no_grad():
    for i in range(num_samples):
        #gen_patch = []
        print(f"### Index {i+args.number} generating")
        for j in range(27):
            print(f"### patch {j} generating")
            cond = torch.tensor([1.0], dtype=torch.float16).to(unet3d.device)
            patch_position = torch.tensor(j, dtype=torch.long).unsqueeze(0).to(unet3d.device)

            # low_res_guidance = get_patch_by_index(guide_dict["pixel_values"], j, mapping) # (1,76,64,64)
            # low_res_guidance = gaussian_blur_3d(low_res_guidance)
            # low_res_guidance = low_res_guidance.unsqueeze(0)
            low_res_guidance = guide_dict["pixel_values"].unsqueeze(0)
            
            image = noise_scheduler.sample(
                vae=vae_model,
                unet=unet3d_model,
                image_size=int(input_size[1] / vae.config.downsample[1]),
                num_frames=int(input_size[0] / vae.config.downsample[0]),
                channels=int(vae.config.embedding_dim),
                patch_position=patch_position, ###
                low_res_guidance=low_res_guidance, ###
                cond=cond, ###
                cond_scale=2., ###
                batch_size=1
            )
            #gen_patch.append(image)
            torch.save(image.cpu(), f"{output_dir}/E8_encoder_patch_{i+args.number}_{j}_gen1.pth")

        #gen_images.append(gen_patch)
