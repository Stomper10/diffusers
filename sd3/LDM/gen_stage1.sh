#!/bin/bash

#SBATCH --job-name=A
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=P2
#SBATCH --time=0-12:00:00
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=8
#SBATCH --signal=B:SIGUSR1@30
#SBATCH --open-mode=append
#SBATCH -o /shared/s1/lab06/wonyoung/diffusers/sd3/LDM/outputs/%x-%j.txt

source /home/s1/wonyoungjang/.bashrc
source /home/s1/wonyoungjang/anaconda3/bin/activate
conda activate diff

echo "Generating stage1."
echo "'c': (3, 1, 0, 2)"

# best /shared/s1/lab06/wonyoung/diffusers/sd3/LDM/results/E2_UNET3D_128/checkpoint-165000
# best /shared/s1/lab06/wonyoung/diffusers/sd3/LDM/results/E2_UNET3D_128/checkpoint-170000
# good /shared/s1/lab06/wonyoung/diffusers/sd3/LDM/results/E2_UNET3D_128/checkpoint-100000

python3 /shared/s1/lab06/wonyoung/diffusers/sd3/LDM/gen_stage1.py \
    --output_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM/gen_volumes/stage1" \
    --pretrained_vae_path="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM/results/E2_VQGAN3D_128/checkpoint-220000" \
    --pretrained_unet_path="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM/results/E2_UNET3D_128/checkpoint-100000" \
    --age=60 \
    --num_samples=1 \
    --separator=0 \
    --resolution="128,64,64" \
    --scheduler="DDPM"

# python3 /shared/s1/lab06/wonyoung/diffusers/sd3/LDM/gen_stage1.py \
#     --output_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM/gen_volumes/stage1" \
#     --pretrained_vae_path="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM/results/E2_VQGAN3D_128/checkpoint-220000" \
#     --pretrained_unet_path="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM/results/E2_UNET3D_128/checkpoint-100000" \
#     --age=80 \
#     --num_samples=100 \
#     --separator=450 \
#     --resolution="128,64,64" 
