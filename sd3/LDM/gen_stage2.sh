#!/bin/bash

#SBATCH --job-name=A7576_s2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=P1
#SBATCH --time=0-12:00:00
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=8
#SBATCH --signal=B:SIGUSR1@30
#SBATCH --open-mode=append
#SBATCH -o /shared/s1/lab06/wonyoung/diffusers/sd3/LDM/outputs/%x-%j.txt

source /home/s1/wonyoungjang/.bashrc
source /home/s1/wonyoungjang/anaconda3/bin/activate
conda activate diff

echo "Generating stage2."
echo "'c': (3, 1, 0, 2)"

python3 /shared/s1/lab06/wonyoung/diffusers/sd3/LDM/gen_stage2.py \
    --output_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM/gen_volumes/stage2" \
    --stage1_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM/gen_volumes/stage1" \
    --pretrained_sr_path="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM/results/E2_SR_128/checkpoint-90000" \
    --resolution="128,128,128" \
    --age=75

python3 /shared/s1/lab06/wonyoung/diffusers/sd3/LDM/gen_stage2.py \
    --output_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM/gen_volumes/stage2" \
    --stage1_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM/gen_volumes/stage1" \
    --pretrained_sr_path="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM/results/E2_SR_128/checkpoint-90000" \
    --resolution="128,128,128" \
    --age=76
