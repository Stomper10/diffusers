import os
from PIL import Image
from accelerate import Accelerator
import torch
import torch.nn.functional as F
from diffusers.models.vq_gan_3d import VQGAN
from diffusers.models.ddpm import PatchUnet3D, PatchGaussianDiffusion
import matplotlib.pyplot as plt

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

pretrained_vae_path="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/E1_pLDM_VQGAN3D/checkpoint-770000"
pretrained_unet_path="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/E2_pLDM_UNET3D_concat/checkpoint-80000"
data_dir="/shared/s1/lab06/20252_individual_samples"
train_label_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/data/ukbb_cn_train.csv"
valid_label_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/data/ukbb_cn_valid.csv"
output_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/patch_generation/concat_500"
axis="c"
seed=42
mixed_precision="fp16"
dataloader_num_workers=4
resolution="76,64,64"
valid_batch_size=1
#num_samples=1
num_timesteps=1000
input_size = tuple(int(x) for x in resolution.split(","))

accelerator = Accelerator(mixed_precision=mixed_precision,)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight_dtype = torch.float16

vae = VQGAN.from_pretrained(pretrained_vae_path, subfolder="vqgan",).to(device)
unet3d = PatchUnet3D.from_pretrained(pretrained_unet_path, subfolder="unet3d",).to(device)
vae.eval()
unet3d.eval()
noise_scheduler = PatchGaussianDiffusion(timesteps=1000,).to(device) ###args

unet3d, vae = accelerator.prepare(unet3d, vae)
vae_model = accelerator.unwrap_model(vae)
unet3d_model = accelerator.unwrap_model(unet3d)


temp_dir = "/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/patch_generation/concat_tmp_slices"
os.makedirs(temp_dir, exist_ok=True)
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
gen_images = []
num_samples = 500
with torch.no_grad():
    for i in range(num_samples):
        gen_patch = []
        for j in range(27):
            print(f"### patch {j} generating")
            
            cond = torch.tensor([1.0], dtype=torch.float16).to(unet3d.device)
            patch_position = torch.tensor(j, dtype=torch.long).unsqueeze(0).to(unet3d.device)
            
            #raw_image = np.load("/shared/s1/lab06/20252_individual_samples/final_array_128_full_7075.npy") #.squeeze(0) # (1,76,64,64)
            image = torch.load("/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_w/results/low-res_gen/tensor_low-res_image_temp.pth").to(torch.float64).to(unet3d.device)
            print(image.shape) # 1,1,76,64,64
            # #image = torch.from_numpy(raw_image).to(torch.float64).to(unet3d.device)  # Stays on CPU
            # axes_mapping = {
            #     's': (3, 0, 1, 2),
            #     'c': (3, 1, 0, 2),
            #     'a': (3, 2, 1, 0)
            # }

            # try:
            #     image = image.permute(*axes_mapping["c"])  # (1,128,128,128)
            # except KeyError:
            #     raise ValueError("axis must be one of 'a', 'c', or 's'.")

            #image = image.unsqueeze(0)
            
            #low_res_guidance = torch.load("/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_w/results/low-res_gen/tensor_low-res_image_origin.pth").to(torch.float64).to(unet3d.device) # (1,1,76,64,64)
            low_res_guidance = F.interpolate( # (1,1,218,182,182)
                image,
                size=(218, 182, 182),
                mode='trilinear',
                align_corners=False
            )
            
            # low_res_guidance = torch.load("/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_w/results/low-res_gen/tensor_low-res_image_origin.pth").to(torch.float64).to(unet3d.device) # (1,1,76,64,64)
            # low_res_guidance = F.interpolate( # (1,1,218,182,182)
            #     low_res_guidance,
            #     size=(218, 182, 182),
            #     mode='trilinear',
            #     align_corners=False
            # )
            low_res_guidance = low_res_guidance.squeeze(0) # (1,218,182,182)
            low_res_guidance = get_patch_by_index(low_res_guidance, j, mapping) # (1,76,64,64)
            low_res_guidance = F.interpolate(
                low_res_guidance.unsqueeze(0).to(torch.float64),
                size=(38, 32, 32),
                mode='trilinear',
                align_corners=False
            ).to(torch.float16)
            
            # print("### batch['condition'].shape:", cond.shape)
            # print("### batch['condition']:", cond)
            # print("### batch['patch_position'].shape:", patch_position.shape)
            # print("### batch['patch_position']:", cond)
            
            image = noise_scheduler.sample(vae=vae_model,
                                        unet=unet3d_model,
                                        image_size=int(input_size[1] / vae.config.downsample[1]),
                                        num_frames=int(input_size[0] / vae.config.downsample[0]),
                                        channels=int(vae.config.embedding_dim),
                                        patch_position=patch_position, ###
                                        low_res_guidance=low_res_guidance, ###
                                        cond=cond, ###
                                        batch_size=1
                                        )
            gen_patch.append(image)
            #gen_slice_images = save_slices_as_images(image.cpu().squeeze().numpy())
            #gen_slice_images[0].save(f"{output_dir}/concat_patch_{j}_gen1.gif", save_all=True, append_images=gen_slice_images[1:], duration=200, loop=0)
            
            torch.save(image.cpu(), f"{output_dir}/concat_patch_{i}_{j}_gen1_tensor_temp.pth")
        gen_images.append(gen_patch)


for img_file in os.listdir(temp_dir):
    os.remove(os.path.join(temp_dir, img_file))
os.rmdir(temp_dir)
# #output_numpy = torch.cat(images, dim=0).numpy()
# print("Inference complete. The output is a 3D NumPy array with shape:", gen_images[0].cpu().squeeze().numpy().shape)


# # Generate sample 3D volume data (replace this with your actual MRI volume data)
# #volume_data_cdhw = np.random.rand(1, 64, 64, 64)  # Replace with your actual 3D volume
# raw_volume_data = raw_images[0].cpu().squeeze().numpy() # (76,64,64)
# recon_volume_data = recon_images[0].cpu().squeeze().numpy() # (76,64,64)
# gen_volume_data = gen_images[0].cpu().squeeze().numpy() # (76,64,64)
# # Directory to store the temporary images
# os.makedirs('tmp_slices', exist_ok=True)

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

# # Create the list of images for each slice
# raw_slice_images = save_slices_as_images(raw_volume_data)
# recon_slice_images = save_slices_as_images(recon_volume_data)
# gen_slice_images = save_slices_as_images(gen_volume_data)

# # Save all the slices as a GIF
# raw_slice_images[0].save(f"{output_dir}/low-res_coronal_raw.gif", save_all=True, append_images=raw_slice_images[1:], duration=200, loop=0)
# recon_slice_images[0].save(f"{output_dir}/low-res_coronal_recon.gif", save_all=True, append_images=recon_slice_images[1:], duration=200, loop=0)
# gen_slice_images[0].save(f"{output_dir}/low-res_coronal_gen.gif", save_all=True, append_images=gen_slice_images[1:], duration=200, loop=0)

# # Cleanup the temporary image files
# for img_file in os.listdir('tmp_slices'):
#     os.remove(os.path.join('tmp_slices', img_file))
# os.rmdir('tmp_slices')

# print(f"GIF saved as {output_dir}/low-res_coronal_raw.gif")
# print(f"GIF saved as {output_dir}/low-res_coronal_recon.gif")
# print(f"GIF saved as {output_dir}/low-res_coronal_gen.gif")
