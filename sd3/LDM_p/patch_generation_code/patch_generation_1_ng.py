import argparse
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers.models.vq_gan_3d import VQGAN
from diffusers.models.ddpm import PatchUnet3D, PatchGaussianDiffusion

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--number", "-n", type=int, required=True, default=0, help="Generation number set.")
args = parser.parse_args()

pretrained_vae_path="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/E1_pLDM_VQGAN3D/checkpoint-770000"
pretrained_unet_path="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/E3_pLDM_UNET3D_noguide/checkpoint-600000"
output_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/patch_generation/E3_Gens_noguide"
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

with torch.no_grad():
    for i in range(num_samples):
        #gen_patch = []
        print(f"### Index {i+args.number} generating")
        for j in range(27):
            print(f"### patch {j} generating")
            cond = torch.tensor([1.0], dtype=torch.float16).to(unet3d.device)
            patch_position = torch.tensor(j, dtype=torch.long).unsqueeze(0).to(unet3d.device)
            
            image = noise_scheduler.sample(
                vae=vae_model,
                unet=unet3d_model,
                image_size=int(input_size[1] / vae.config.downsample[1]),
                num_frames=int(input_size[0] / vae.config.downsample[0]),
                channels=int(vae.config.embedding_dim),
                patch_position=patch_position, ###
                low_res_guidance=None, ###
                cond=cond, ###
                cond_scale=2., ###
                batch_size=1
            )
            #gen_patch.append(image)
            torch.save(image.cpu(), f"{output_dir}/E3_noguide_patch_{i+args.number}_{j}_gen1.pth")

        #gen_images.append(gen_patch)
