import os
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import accelerate
from accelerate import Accelerator

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from monai import transforms

from diffusers.models.vq_gan_3d import VQGAN
from diffusers.models.ddpm import PatchUnet3D, PatchGaussianDiffusion
#from mayavi import mlab

import numpy as np
import matplotlib.pyplot as plt
# import ipywidgets as widgets
# from ipywidgets import interact
# from sklearn.manifold import TSNE

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


class UKB_Dataset(Dataset):
    """
    A PyTorch Dataset for loading UKB images and labels.
    
    Args:
        image_dir (str): Directory containing image files.
        label_dir (str): CSV file containing image IDs and labels.
        transform (callable, optional): Optional transform to apply to samples.
        axis (str): Axis to permute ('a', 'c', or 's').
    """
    def __init__(self, image_dir, label_dir, transform=None, axis="s", patch_position=0):
        super().__init__()
        self.data_dir = image_dir
        data_csv = pd.read_csv(label_dir)
        self.image_names = list(data_csv['id'])

        # load conditioning variable: age
        self.ages = data_csv['age'].values.astype(np.float16)
        self.min_age, self.max_age = self.ages.min(), self.ages.max()
        self.norm_ages = (self.ages - self.min_age) / (self.max_age - self.min_age)

        # # load conditioning variable: ventricular volume
        # self.bvvs = data_csv['ventricular_volume'].values.astype(np.float32)
        # self.min_bvv, self.max_bvv = self.bvvs.min(), self.bvvs.max()
        # self.norm_bvvs = (self.bvvs - self.min_bvv) / (self.max_bvv - self.min_bvv)

        # # load conditioning variable: gender
        # self.genders = data_csv['gender'].values
        # self.gender_encoded = []
        # for gender_str in self.genders:
        #     if gender_str == 'Male':
        #         self.gender_encoded.append([1.0, 0.0])
        #     elif gender_str == 'Female':
        #         self.gender_encoded.append([0.0, 1.0])
        #     else:
        #         self.gender_encoded.append([0.0, 0.0])  # Handle unknown gender
        # self.gender_encoded = np.array(self.gender_encoded, dtype=np.float16)

        self.patch_position = patch_position
        self.transform = transform
        self.axis = axis
        self.image_paths = [
            os.path.join(self.data_dir, f'final_array_128_full_{name}.npy')
            for name in self.image_names
        ]

        # Initialize lists
        depth_starts = [0, 71, 142]
        height_starts = [0, 59, 118]
        width_starts = [0, 59, 118]

        # Initialize index
        index = 0
        # Mapping list to store results
        self.mapping = []

        for id, start_d in enumerate(depth_starts):
            for ih, start_h in enumerate(height_starts):
                for iw, start_w in enumerate(width_starts):
                    self.mapping.append({
                        'index': index,
                        'start_d': start_d,
                        'start_h': start_h,
                        'start_w': start_w
                    })
                    print(f"Index: {index}, Start Coordinates: (D: {start_d}, H: {start_h}, W: {start_w})")
                    index += 1

        # low-res guidance
        #self.lowres_guidance = torch.load("/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_w/results/low-res_gen/tensor_low-res_image_origin.pth").to(torch.float64)

    def get_patch_by_index(self, volume, index, mapping):
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

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        #lowres_guide = self.lowres_guidance

        try:
            image = np.load(image_path)  # (128,128,128,1)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = torch.from_numpy(image)  # Stays on CPU
        
        axes_mapping = {
            's': (3, 0, 1, 2),
            'c': (3, 1, 0, 2),
            'a': (3, 2, 1, 0)
        }

        try:
            image = image.permute(*axes_mapping[self.axis])  # (1,128,128,128)
        except KeyError:
            raise ValueError("axis must be one of 'a', 'c', or 's'.")
        
        # Add batch dimension (N=1) for interpolation
        image = image.unsqueeze(0)  # Shape: (1, 1, D, H, W)

        # Define target size
        target_size = (218, 182, 182)  # (D₂, H₂, W₂)
        lowres_size = (76, 64, 64)

        # Resize the volume using trilinear interpolation
        image = F.interpolate(
            image,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )  # Shape: (1, 1, D₂, H₂, W₂)
        lowres_guide = F.interpolate(
            image,
            size=lowres_size,
            mode='trilinear',
            align_corners=False
        )  # Shape: (1, 1, D₂, H₂, W₂)
        lowres_guide = F.interpolate(
            lowres_guide,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )  # Shape: (1, 1, D₂, H₂, W₂)

        # Remove the batch dimension
        image = image.squeeze(0).to(torch.float16)  # Shape: (1, D₂, H₂, W₂) # (1,218,182,182)
        lowres_guide = lowres_guide.squeeze(0).to(torch.float16)  # Shape: (1, D₂, H₂, W₂) # (1,218,182,182)

        age = torch.tensor([self.norm_ages[index]], dtype=torch.float16) # Shape: [1]
        # gender = torch.tensor(self.gender_encoded[index], dtype=torch.float16)  # Shape: [2]
        # bvv = torch.tensor([self.normalized_bvvs[index]], dtype=torch.float16)  # Shape: [1]

        #cond_tensor = torch.cat([age, gender, bvv], dim=-1)  # Shape: [4]

        sample = {
            "pixel_values": image,
            "condition": age,
            #"lowres_guide": lowres_guide,
        }

        if self.transform:
            sample = self.transform(sample)
        del image

        patch_position_sampled = random.randint(0, 26)
        sample["pixel_values"] = self.get_patch_by_index(sample["pixel_values"], patch_position_sampled, self.mapping)
        sample["lowres_guide"] = self.get_patch_by_index(lowres_guide, patch_position_sampled, self.mapping)
        sample["lowres_guide"] = F.interpolate(
            sample["lowres_guide"].unsqueeze(0).to(torch.float64),
            size=(38, 32, 32),
            mode='trilinear',
            align_corners=False
        ).squeeze(0).to(torch.float16)
        sample["patch_position"] = torch.tensor(patch_position_sampled, dtype=torch.long)

        return sample


pretrained_vae_path="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/E1_pLDM_VQGAN3D/checkpoint-770000"
pretrained_unet_path="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/E2_pLDM_UNET3D_noguide/checkpoint-60000"
data_dir="/shared/s1/lab06/20252_individual_samples"
train_label_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/data/ukbb_cn_train.csv"
valid_label_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/data/ukbb_cn_valid.csv"
output_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/patch_generation/noguide_500"
axis="c"
seed=42
mixed_precision="fp16"
dataloader_num_workers=4
resolution="76,64,64"
valid_batch_size=1
#num_samples=1
num_timesteps=1000
input_size = tuple(int(x) for x in resolution.split(","))

accelerator = Accelerator(
        mixed_precision=mixed_precision,
    )
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight_dtype = torch.float16


# for recon array from VQGAN
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
valid_dataset = UKB_Dataset(data_dir, valid_label_dir, transform=train_transforms, axis=axis)
test_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        shuffle=False,
        #collate_fn=collate_fn,
        batch_size=valid_batch_size,
        num_workers=dataloader_num_workers,
        pin_memory=True
    )

vae = VQGAN.from_pretrained(pretrained_vae_path, subfolder="vqgan",).to(device)
unet3d = PatchUnet3D.from_pretrained(pretrained_unet_path, subfolder="unet3d",).to(device)
vae.eval()
unet3d.eval()
noise_scheduler = PatchGaussianDiffusion( # diffusers pipeline?
        #unet3d,
        #vqgan_ckpt=cfg.model.vqgan_ckpt,
        #image_size=cfg.model.diffusion_img_size,
        #num_frames=cfg.model.diffusion_depth_size,
        #channels=cfg.model.diffusion_num_channels,
        timesteps=1000,
        # sampling_timesteps=cfg.model.sampling_timesteps,
        #loss_type=cfg.model.loss_type,
        # objective=cfg.objective
    ).to(device) ###args

unet3d, vae, test_dataloader = accelerator.prepare(
        unet3d, vae, test_dataloader
    )
vae_model = accelerator.unwrap_model(vae)
unet3d_model = accelerator.unwrap_model(unet3d)


temp_dir = "/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/patch_generation/noguide_tmp_slices"
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


# Reconstruction
# print("Reconstruction starts")
# recon_images = []
# raw_images = []

# with torch.no_grad():
#     for i, batch in enumerate(test_dataloader):
#         if i == num_samples:
#             break

#         patch_recon_images = []
#         patch_raw_images = []
#         for j in range(27):
#             with torch.autocast('cpu', dtype=weight_dtype):
#                 print(f"### patch {j} processing")
#                 x = batch["pixel_values"].to(weight_dtype)
#                 cond = torch.tensor([1.0], dtype=torch.float16).to(unet3d.device)
#                 patch_position = torch.tensor(j, dtype=torch.long).unsqueeze(0).to(unet3d.device)
#                 lowres_guide = torch.load("/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_w/results/low-res_gen/tensor_low-res_image_origin.pth").to(torch.float64).to(unet3d.device)
                
#                 print("### x.shape:", x.shape) # torch.Size([1])
#                 print("### cond.shape:", cond.shape) # tensor([0.5000], device='cuda:0', dtype=torch.float16)
#                 print("### patch_position.shape:", patch_position.shape) # torch.Size([1])
#                 print("### lowres_guide.shape:", lowres_guide.shape) # tensor([0.5000], device='cuda:0', dtype=torch.float16)

#                 x_recon, _ = vae_model(x)

#                 #output_image = x_recon.squeeze(0)
#                 #patch_recon_images.append(x_recon.cpu())
#                 patch_raw_images.append(x.cpu())
#                 #torch.save(x.cpu(), f"{output_dir}/tensor_low-res_image_origin_temp.pth")
#                 #recon_slice_images = save_slices_as_images(x_recon.cpu().squeeze().numpy())
#                 raw_slice_images = save_slices_as_images(x.cpu().squeeze().numpy())

#                 #recon_slice_images[0].save(f"{output_dir}/patch_{j}_recon.gif", save_all=True, append_images=recon_slice_images[1:], duration=200, loop=0)
#                 raw_slice_images[0].save(f"{output_dir}/patch_{j}_raw.gif", save_all=True, append_images=raw_slice_images[1:], duration=200, loop=0)

        
#         recon_images.append(patch_recon_images)
#         raw_images.append(patch_raw_images)



# # #output_numpy = torch.cat(images, dim=0).numpy()
# raw_volume_data = raw_images[0].cpu().squeeze().numpy() # (76,64,64)
# recon_volume_data = recon_images[0].cpu().squeeze().numpy() # (76,64,64)
# raw_slice_images = save_slices_as_images(raw_volume_data)
# recon_slice_images = save_slices_as_images(recon_volume_data)

# print("Reconstruction complete. The output is a 3D NumPy array with shape:", recon_images[0][0].cpu().squeeze().numpy().shape)

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
            # batch = next(iter(test_dataloader))
            # x = batch["pixel_values"][:num_samples]
            # cond = batch["condition"][:num_samples]
            # patch_position = batch["patch_position"][:num_samples]
            # low_res_guidance = batch["lowres_guide"][:num_samples]
            
            cond = torch.tensor([1.0], dtype=torch.float16).to(unet3d.device)
            patch_position = torch.tensor(j, dtype=torch.long).unsqueeze(0).to(unet3d.device)
            # low_res_guidance = torch.load("/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_w/results/low-res_gen/tensor_low-res_image_origin.pth").to(torch.float64).to(unet3d.device) # (1,1,76,64,64)
            # low_res_guidance = F.interpolate( # (1,1,218,182,182)
            #     low_res_guidance,
            #     size=(218, 182, 182),
            #     mode='trilinear',
            #     align_corners=False
            # )
            # low_res_guidance = low_res_guidance.squeeze(0).to(torch.float16) # (1,218,182,182)
            # low_res_guidance = get_patch_by_index(low_res_guidance, j, mapping) # (1,76,64,64)
            # low_res_guidance = F.interpolate(
            #     low_res_guidance.unsqueeze(0).to(torch.float64),
            #     size=(38, 32, 32),
            #     mode='trilinear',
            #     align_corners=False
            # ).to(torch.float16)
            
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
                                        low_res_guidance=None, ###
                                        cond=cond, ###
                                        batch_size=1
                                        )
            gen_patch.append(image)
            # gen_slice_images = save_slices_as_images(image.cpu().squeeze().numpy())
            # gen_slice_images[0].save(f"{output_dir}/noguide_patch_{j}_gen1.gif", save_all=True, append_images=gen_slice_images[1:], duration=200, loop=0)
            
            torch.save(image.cpu(), f"{output_dir}/noguide_patch_{i}_{j}_gen1_tensor.pth")
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
