import os
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
from diffusers.models.ddpm import Unet3D, GaussianDiffusion
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
    def __init__(self, image_dir, label_dir, transform=None, axis="s"):
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
        
        self.transform = transform
        self.axis = axis
        self.image_paths = [
            os.path.join(self.data_dir, f'final_array_128_full_{name}.npy')
            for name in self.image_names
        ]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        
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

        # Resize the volume using trilinear interpolation
        image = F.interpolate(
            image,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )  # Shape: (1, 1, D₂, H₂, W₂)

        # Remove the batch dimension
        image = image.squeeze(0).to(torch.float16)  # Shape: (1, D₂, H₂, W₂) # (1,218,182,182)
        
        age = torch.tensor([self.norm_ages[index]], dtype=torch.float16) # Shape: [1]
        # gender = torch.tensor(self.gender_encoded[index], dtype=torch.float16)  # Shape: [2]
        # bvv = torch.tensor([self.normalized_bvvs[index]], dtype=torch.float16)  # Shape: [1]

        #cond_tensor = torch.cat([age, gender, bvv], dim=-1)  # Shape: [4]

        sample = {
            "pixel_values": image,
            "condition": age #cond_tensor
        }

        if self.transform:
            sample = self.transform(sample)
        del image

        return sample


pretrained_vae_path="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_w/results/E0_wLDM_VQGAN3D/checkpoint-1000000"
pretrained_unet_path="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_w/results/E0_wLDM_UNET3D/checkpoint-480000"
data_dir="/shared/s1/lab06/20252_individual_samples"
train_label_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/data/ukbb_cn_train.csv"
valid_label_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/data/ukbb_cn_valid.csv"
output_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_w/results/low-res_gen"
axis="c"
seed=42
mixed_precision="fp16"
dataloader_num_workers=4
resolution="76,64,64"
valid_batch_size=1
num_samples=1
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
        transforms.Resized(keys=["pixel_values"], spatial_size=input_size, size_mode="all"),
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
        batch_size=1,
        num_workers=dataloader_num_workers,
        pin_memory=True
    )

vae = VQGAN.from_pretrained(pretrained_vae_path, subfolder="vqgan",).to(device)
unet3d = Unet3D.from_pretrained(pretrained_unet_path, subfolder="unet3d",).to(device)
vae.eval()
unet3d.eval()
noise_scheduler = GaussianDiffusion( # diffusers pipeline?
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

# Reconstruction
print("Reconstruction starts")
recon_images = []
raw_images = []
with torch.no_grad():
    for i, batch in enumerate(test_dataloader):
        if i == num_samples:
            break

        with torch.autocast('cpu', dtype=weight_dtype):
            x = batch["pixel_values"].to(weight_dtype)
            x_recon, _ = vae_model(x)

            #output_image = x_recon.squeeze(0)
            recon_images.append(x_recon.cpu())
            raw_images.append(x.cpu())
            torch.save(x.cpu(), f"{output_dir}/tensor_low-res_image_origin_temp.pth")
            
#output_numpy = torch.cat(images, dim=0).numpy()
print("Reconstruction complete. The output is a 3D NumPy array with shape:", recon_images[0].cpu().squeeze().numpy().shape)

# Generation
print("Inference starts")
gen_images = []
with torch.no_grad():
    for i in range(num_samples):
        cond = torch.tensor([0.5], dtype=torch.float16).to(unet3d.device)
        image = noise_scheduler.sample(vae=vae_model,
                                       unet=unet3d_model,
                                       image_size=int(input_size[1] / vae.config.downsample[1]),
                                       num_frames=int(input_size[0] / vae.config.downsample[0]),
                                       channels=int(vae.config.embedding_dim),
                                       cond=cond,
                                       batch_size=1
                                       )
        gen_images.append(image)
        torch.save(image.cpu(), f"{output_dir}/tensor_low-res_image_temp.pth")

#output_numpy = torch.cat(images, dim=0).numpy()
print("Inference complete. The output is a 3D NumPy array with shape:", gen_images[0].cpu().squeeze().numpy().shape)


# Generate sample 3D volume data (replace this with your actual MRI volume data)
#volume_data_cdhw = np.random.rand(1, 64, 64, 64)  # Replace with your actual 3D volume
raw_volume_data = raw_images[0].cpu().squeeze().numpy() # (76,64,64)
recon_volume_data = recon_images[0].cpu().squeeze().numpy() # (76,64,64)
gen_volume_data = gen_images[0].cpu().squeeze().numpy() # (76,64,64)
# Directory to store the temporary images
os.makedirs('tmp_slices', exist_ok=True)

# Generate and save each 2D slice as an image
def save_slices_as_images(volume_data):
    slice_images = []
    for i in range(volume_data.shape[0]): ### c:0, s:1, a:2
        plt.figure(figsize=(7, 7))
        plt.imshow(volume_data[i, :, :], cmap='gray') ### c, s, a
        plt.title(f'Slice {i}')
        plt.axis('off')
        
        # Save each slice to a temporary file
        file_name = f'tmp_slices/slice_{i}.png'
        plt.savefig(file_name)
        plt.close()
        
        # Open the image and append to the list
        slice_images.append(Image.open(file_name))
    
    return slice_images

# Create the list of images for each slice
raw_slice_images = save_slices_as_images(raw_volume_data)
recon_slice_images = save_slices_as_images(recon_volume_data)
gen_slice_images = save_slices_as_images(gen_volume_data)

# Save all the slices as a GIF
raw_slice_images[0].save(f"{output_dir}/low-res_coronal_raw.gif", save_all=True, append_images=raw_slice_images[1:], duration=200, loop=0)
recon_slice_images[0].save(f"{output_dir}/low-res_coronal_recon.gif", save_all=True, append_images=recon_slice_images[1:], duration=200, loop=0)
gen_slice_images[0].save(f"{output_dir}/low-res_coronal_gen.gif", save_all=True, append_images=gen_slice_images[1:], duration=200, loop=0)

# Cleanup the temporary image files
for img_file in os.listdir('tmp_slices'):
    os.remove(os.path.join('tmp_slices', img_file))
os.rmdir('tmp_slices')

print(f"GIF saved as {output_dir}/low-res_coronal_raw.gif")
print(f"GIF saved as {output_dir}/low-res_coronal_recon.gif")
print(f"GIF saved as {output_dir}/low-res_coronal_gen.gif")
