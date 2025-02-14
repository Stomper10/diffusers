import os
import argparse
import numpy as np
from monai import transforms
from accelerate import Accelerator
import torch
from diffusers.models.vq_gan_3d import SRUNET3D

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--output_dir", type=str, default=None, required=True, help="output dir where files saved.")
parser.add_argument("--stage1_dir", type=str, default=None, required=True, help="stage1 dir where files be upscaled.")
parser.add_argument("--pretrained_sr_path", type=str, default=None, required=True, help="SRUNET3D path.")
parser.add_argument("--resolution", "-r", type=str, required=True, default=0, help="SR resolution.")
parser.add_argument("--age", "-a", type=int, required=True, default=0, help="Generation age.") # SET
#parser.add_argument("--age", "-a", type=int, required=True, default=0, help="SR age.") # SET
args = parser.parse_args()

print(f"# resolution: {args.resolution}")

resolution = args.resolution #"128,128,128" #"224,160,160"
input_size = tuple(int(x) for x in resolution.split(","))
output_dir = args.output_dir
pretrained_sr_path = args.pretrained_sr_path
#stage1_list = os.listdir(args.stage1_dir)
stage1_list = [file for file in os.listdir(args.stage1_dir) if f"A{args.age}" in file] # SET
print(f"### A{args.age}: {len(stage1_list)}")

mixed_precision="fp16"
accelerator = Accelerator(mixed_precision=mixed_precision)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

srmodel = SRUNET3D.from_pretrained(pretrained_sr_path, subfolder="srmodel",).to(device)
srmodel.eval()
srmodel = accelerator.prepare(srmodel)
srmodel = accelerator.unwrap_model(srmodel)

train_transforms = transforms.Compose(
        [
            transforms.ScaleIntensityd(keys=["lr"], minv=-1.0, maxv=1.0),
            transforms.ThresholdIntensityd(keys=["lr"], threshold=1, above=False, cval=1.0),
            transforms.ThresholdIntensityd(keys=["lr"], threshold=-1, above=True, cval=-1.0),
            transforms.ToTensord(keys=["lr"]),
        ]
    )

# Hanning window !!!!! hanning after SR is much better than hanning before SR, so use here.
D_p, H_p, W_p = input_size
w_d = torch.hann_window(D_p, periodic=False, dtype=torch.float32)
w_h = torch.hann_window(H_p, periodic=False, dtype=torch.float32)
w_w = torch.hann_window(W_p, periodic=False, dtype=torch.float32)

weight_mask = w_d[:, None, None] * w_h[None, :, None] * w_w[None, None, :]
weight_mask = weight_mask / weight_mask.max()
weight_mask = weight_mask.unsqueeze(0).to(srmodel.device) # (1,128,128,128)

# Upscaling
lr_dict = dict()
print("Upscaling starts.")
#stage1_list=["N0_stage1_A65.pth"] # SET
with torch.no_grad():
    for i, name in enumerate(stage1_list):
        print(f"### Index {i} upscaling")
        new_name = name.replace("stage1", "stage2").replace("pth", "npy")
        print(new_name)

        lr = torch.load(f"{args.stage1_dir}/{name}").to(torch.float16).to(srmodel.device) # (1,128,64,64)
        lr_dict["lr"] = lr # raw
        lr_dict = train_transforms(lr_dict)
        image0 = srmodel(lr_dict["lr"].unsqueeze(0)) # (1,1,128,128,128)
        #torch.save(image0.squeeze(0).cpu(), f"{output_dir}/img0_{new_name}.pth")
        np.save(f"{output_dir}/img0_{new_name}", image0.squeeze(0).cpu().numpy()) # (1,128,128,128)

        # Create the 3D Hann window for the given dimensions
        weighted_whole = image0.squeeze(0) * weight_mask
        weight_mask[weight_mask == 0] = 1
        image1 = weighted_whole / weight_mask
        #torch.save(image1.cpu(), f"{output_dir}/img1_{new_name}.pth") 
        np.save(f"{output_dir}/img1_{new_name}", image1.cpu().numpy()) # (1,128,128,128)

#output_numpy = torch.cat(images, dim=0).numpy()
print("Inference complete. The output is a 3D NumPy array with shape:", image1.cpu().numpy().shape)



# # Generate GIF
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

# gif_list = [image0, image1] #, image2]
# output_dir_gif="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM/gen_volumes/gif"
# new_name_gif=new_name.replace("npy", "gif")
# for i, vol in enumerate(gif_list):
#     os.makedirs('tmp_slices', exist_ok=True)
#     gen_volume_data = vol.cpu().squeeze().numpy() #
#     gen_slice_images = save_slices_as_images(gen_volume_data)
#     gen_slice_images[0].save(f"{output_dir_gif}/img{i}_{new_name_gif}", save_all=True, append_images=gen_slice_images[1:], duration=200, loop=0)

#     # Cleanup the temporary image files
#     for img_file in os.listdir('tmp_slices'):
#         os.remove(os.path.join('tmp_slices', img_file))
#     os.rmdir('tmp_slices')
#     print(f"GIF saved as {output_dir_gif}/img{i}_{new_name_gif}")
