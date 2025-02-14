import argparse
from accelerate import Accelerator
import torch
from diffusers.models.vq_gan_3d import VQGAN
from diffusers.models.ddpm import Unet3D, GaussianDiffusion, DDIMScheduler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--output_dir", type=str, default=None, required=True, help="output dir where files saved.")
parser.add_argument("--pretrained_vae_path", type=str, default=None, required=True, help="VQGAN path.")
parser.add_argument("--pretrained_unet_path", type=str, default=None, required=True, help="Unet3D path.")
parser.add_argument("--age", "-a", type=int, required=True, default=0, help="Generation age.") # SET
parser.add_argument("--num_samples", "-n", type=int, required=True, default=0, help="Generation number set.") # SET
parser.add_argument("--separator", "-s", type=int, required=True, default=0, help="Generation number set.") # SET
parser.add_argument("--resolution", "-r", type=str, required=True, default=0, help="Generation resolution.") # SET
parser.add_argument("--scheduler", type=str, required=True, default="DDPM", help="Generation scheduler.") # SET
args = parser.parse_args()

print(f"# age: {args.age}")
print(f"# num_samples: {args.num_samples}")
print(f"# resolution: {args.resolution}")

norm_age = (args.age - 45) / (81 - 45)
num_samples = args.num_samples # SET 200
resolution = args.resolution # SET #"128,64,64" "224,40,40"
input_size = tuple(int(x) for x in resolution.split(","))

output_dir = args.output_dir # SET /stage1
pretrained_vae_path = args.pretrained_vae_path
pretrained_unet_path = args.pretrained_unet_path
# pretrained_vae_path="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM/results/E2_VQGAN3D_128/checkpoint-220000" # SET
# pretrained_unet_path="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM/results/E2_UNET3D_128/checkpoint-70000"  # SET
# pretrained_vae_path="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM/results/E2_VQGAN3D_160/checkpoint-530000" # SET
# pretrained_unet_path="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM/results/E2_UNET3D_160/checkpoint-70000"  # SET

mixed_precision = "fp16"
accelerator = Accelerator(mixed_precision=mixed_precision)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae = VQGAN.from_pretrained(pretrained_vae_path, subfolder="vqgan",).to(device)
unet3d = Unet3D.from_pretrained(pretrained_unet_path, subfolder="unet3d",).to(device)
vae.eval()
unet3d.eval()
if args.scheduler == "DDPM":
    noise_scheduler = GaussianDiffusion(timesteps=1000).to(device)
elif args.scheduler == "DDIM":
    noise_scheduler = DDIMScheduler(timesteps=200).to(device)

unet3d, vae, = accelerator.prepare(unet3d, vae)
vae_model = accelerator.unwrap_model(vae)
unet3d_model = accelerator.unwrap_model(unet3d)

# Hanning window !!!!! hanning after SR is much better than hanning before SR, so don't use here.
# D_p, H_p, W_p = input_size
# w_d = torch.hann_window(D_p, periodic=False, dtype=torch.float32)
# w_h = torch.hann_window(H_p, periodic=False, dtype=torch.float32)
# w_w = torch.hann_window(W_p, periodic=False, dtype=torch.float32)

# weight_mask = w_d[:, None, None] * w_h[None, :, None] * w_w[None, None, :]
# weight_mask = weight_mask / weight_mask.max()
# weight_mask = weight_mask.unsqueeze(0).to(unet3d.device) # (1,128,64,64)

# Generation
print("Inference starts")
with torch.no_grad():
    for i in range(num_samples):
        print(f"### Index {i+args.separator} generating")
        age = torch.tensor([norm_age], dtype=torch.float16).to(unet3d.device)
        sex = torch.randint(0,2,(1,), dtype=torch.float16).to(unet3d.device) # torch.tensor([1.0], dtype=torch.float16).to(unet3d.device) #
        vcf = torch.rand(1, dtype=torch.float16).to(unet3d.device) # torch.tensor([1.0], dtype=torch.float16).to(unet3d.device)
        bv = torch.rand(1, dtype=torch.float16).to(unet3d.device) # torch.tensor([0.5], dtype=torch.float16).to(unet3d.device) #
        cond_tensor = torch.cat([age, sex, vcf, bv], dim=-1) # Shape: [4]

        image0 = noise_scheduler.sample(vae=vae_model, # (1,1,128,64,64)
                                       unet=unet3d_model,
                                       image_size=int(input_size[1] / vae.config.downsample[1]),
                                       num_frames=int(input_size[0] / vae.config.downsample[0]),
                                       channels=int(vae.config.embedding_dim),
                                       cond=cond_tensor,
                                       cond_scale=2., ### 2.
                                       batch_size=1
                                       )
        torch.save(image0.squeeze(0).cpu(), f"{output_dir}/N{i+args.separator}_stage1_A{args.age}.pth") # (1,128,64,64)
        
        # Suppose you have a tensor named vol of shape (1, 1, 240, 160, 160)
        # Create the 3D Hann window for the given dimensions
        # weighted_whole = image0.squeeze(0) * weight_mask
        # weight_mask[weight_mask == 0] = 1
        # image1 = weighted_whole / weight_mask
        # torch.save(image1.cpu(), f"{output_dir}/N{i+args.num_samples-1}_stage1_img1_A{args.age}.pth") # (1,128,64,64)

#output_numpy = torch.cat(images, dim=0).numpy()
print("Inference complete. The output is a 3D NumPy array with shape:", image0.cpu().numpy().shape)



# # Generate GIF
# import os
# from PIL import Image
# import matplotlib.pyplot as plt

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

# gif_list = [image0] #, image1] #, image2]
# output_dir_gif="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM/gen_volumes/gif" # SET /stage1
# for i, vol in enumerate(gif_list):
#     os.makedirs('tmp_slices', exist_ok=True)
#     gen_volume_data = vol.cpu().squeeze().numpy() #
#     gen_slice_images = save_slices_as_images(gen_volume_data)
#     gen_slice_images[0].save(f"{output_dir_gif}/N0_stage1_A{args.age}.gif", save_all=True, append_images=gen_slice_images[1:], duration=200, loop=0)

#     # Cleanup the temporary image files
#     for img_file in os.listdir('tmp_slices'):
#         os.remove(os.path.join('tmp_slices', img_file))
#     os.rmdir('tmp_slices')
#     print(f"GIF saved as {output_dir_gif}/N0_stage1_A{args.age}.gif")
