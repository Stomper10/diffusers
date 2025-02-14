import os
import argparse
from PIL import Image
from accelerate import Accelerator
import torch
import torch.nn.functional as F
from diffusers.models.vq_gan_3d import VQGAN
from diffusers.models.ddpm import Unet3D, GaussianDiffusion
import matplotlib.pyplot as plt

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


# def gaussian_3d_kernel(kernel_size: int, sigma: float, device='cpu', dtype=torch.float64):
#     """
#     Create a 3D Gaussian kernel for convolution-based blurring.
#     The returned kernel shape will be (1,1,kernel_size,kernel_size,kernel_size).
#     """
#     # Coordinates around 0
#     coords = torch.arange(kernel_size, dtype=dtype, device=device) - (kernel_size - 1) / 2
#     z, y, x = torch.meshgrid(coords, coords, coords, indexing='ij')
#     g = torch.exp(-(x**2 + y**2 + z**2) / (2.0 * sigma**2))
#     g /= g.sum()  # Normalize so that sum of kernel = 1
#     # Reshape to fit conv3d expected shape: (out_channels, in_channels, kD, kH, kW)
#     g = g.unsqueeze(0).unsqueeze(0)
#     return g

# def unsharp_mask_3d(
#     x: torch.Tensor,
#     alpha: float = 0.5,
#     kernel_size: int = 3,
#     sigma: float = 1.0
# ) -> torch.Tensor:
#     """
#     Apply 3D unsharp masking to volume x.
#     x shape: (N, C, D, H, W).
#     1) Blur with a Gaussian kernel.
#     2) mask = original - blurred.
#     3) sharpened = original + alpha * mask.
#     """
#     device, dtype = x.device, x.dtype
#     # Create 3D Gaussian blur kernel
#     kernel = gaussian_3d_kernel(kernel_size, sigma, device=device, dtype=dtype)
#     padding = kernel_size // 2

#     # Blur the volume
#     blurred = F.conv3d(x, kernel, padding=padding)

#     # High-frequency content
#     mask = x - blurred

#     # Add scaled mask to original
#     sharpened = x + alpha * mask
#     return sharpened


parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--age", "-a", type=int, required=True, default=0, help="Generation age.") # SET
parser.add_argument("--number", "-n", type=int, required=True, default=0, help="Generation number set.") # SET
args = parser.parse_args()

#args.number=0 # SET
NUM_SAMPLES=120 # SET 200
output_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_w/results/stage1" # SET /stage1
norm_age = (args.age - 44) / 33

pretrained_vae_path="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_w/results/E7_wLDM_VQGAN3D/checkpoint-440000"
pretrained_unet_path="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_w/results/E7_wLDM_UNET3D/checkpoint-109000"
mixed_precision="fp16"
dataloader_num_workers=4
resolution="224,40,40"
input_size = tuple(int(x) for x in resolution.split(","))

accelerator = Accelerator(mixed_precision=mixed_precision)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight_dtype = torch.float16

vae = VQGAN.from_pretrained(pretrained_vae_path, subfolder="vqgan",).to(device)
unet3d = Unet3D.from_pretrained(pretrained_unet_path, subfolder="unet3d",).to(device)
vae.eval()
unet3d.eval()
noise_scheduler = GaussianDiffusion(timesteps=1000).to(device) ###args

unet3d, vae, = accelerator.prepare(unet3d, vae)
vae_model = accelerator.unwrap_model(vae)
unet3d_model = accelerator.unwrap_model(unet3d)

# Suppose your input tensor is named 'volume' with shape [1,1,224,160,160]
D_p, H_p, W_p = (224, 40, 40)
w_d = torch.hann_window(D_p, periodic=False, dtype=torch.float32)
w_h = torch.hann_window(H_p, periodic=False, dtype=torch.float32)
w_w = torch.hann_window(W_p, periodic=False, dtype=torch.float32)

weight_mask = w_d[:, None, None] * w_h[None, :, None] * w_w[None, None, :]
weight_mask = weight_mask / weight_mask.max()
weight_mask = weight_mask.unsqueeze(0)  # (1, D_p, H_p, W_p)

# Generation
print("Inference starts")
with torch.no_grad():
    for i in range(NUM_SAMPLES):
        print(f"### Index {i+args.number} generating")
        cond = torch.tensor([norm_age], dtype=torch.float16).to(unet3d.device)
        image0 = noise_scheduler.sample(vae=vae_model,
                                       unet=unet3d_model,
                                       image_size=int(input_size[1] / vae.config.downsample[1]),
                                       num_frames=int(input_size[0] / vae.config.downsample[0]),
                                       channels=int(vae.config.embedding_dim),
                                       cond=cond,
                                       cond_scale=2., ###
                                       batch_size=1
                                       )
        print(image0.shape)
        torch.save(image0.cpu(), f"{output_dir}/{i+args.number}_stage1_{args.age}.pth") # SET _justgen
        
        # # Suppose you have a tensor named vol of shape (1, 1, 240, 160, 160)
        # image1 = unsharp_mask_3d(image0.cpu(), alpha=0.3, kernel_size=5, sigma=1.0)
        # print(image1.shape)
        # torch.save(image1.cpu(), f"{output_dir}/{i+args.number}_stage1_sharp.pth")

        # # Create the 3D Hann window for the given dimensions [224, 160, 160]
        # weighted_whole = image1 * weight_mask
        # weight_mask[weight_mask == 0] = 1
        # image2 = weighted_whole / weight_mask
        # print(image2.shape)
        # torch.save(image2.cpu(), f"{output_dir}/{i+args.number}_stage1_sharphann.pth")

#output_numpy = torch.cat(images, dim=0).numpy()
print("Inference complete. The output is a 3D NumPy array with shape:", image0.cpu().squeeze().numpy().shape)



# # Generate and save each 2D slice as an image
# def save_slices_as_images(volume_data):
#     slice_images = []
#     for i in range(volume_data.shape[0]): ### c:0, s:1, a:2
#         plt.figure(figsize=(7, 7))
#         plt.imshow(volume_data[i, :, :], cmap='gray') ### c, s, a
#         plt.title(f'Slice {i}')
#         plt.axis('off')
        
#         # Save each slice to a temporary file
#         file_name = f'tmp_slices/slice_{i}.png'
#         plt.savefig(file_name)
#         plt.close()
        
#         # Open the image and append to the list
#         slice_images.append(Image.open(file_name))
    
#     return slice_images

# gif_list = [image0, image1, image2]
# for i, vol in enumerate(gif_list):
#     os.makedirs('tmp_slices', exist_ok=True)
#     gen_volume_data = vol.cpu().squeeze().numpy() # (76,64,64)
#     gen_slice_images = save_slices_as_images(gen_volume_data)
#     gen_slice_images[0].save(f"{output_dir}/stage1_{i}.gif", save_all=True, append_images=gen_slice_images[1:], duration=200, loop=0)

#     # Cleanup the temporary image files
#     for img_file in os.listdir('tmp_slices'):
#         os.remove(os.path.join('tmp_slices', img_file))
#     os.rmdir('tmp_slices')
#     print(f"GIF saved as {output_dir}/stage1_{i}.gif")
