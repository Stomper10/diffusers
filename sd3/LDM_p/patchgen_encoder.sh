#!/bin/bash

#SBATCH --job-name=E8_patchgen_encoder_0
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=P2
#SBATCH --time=0-12:00:00
#SBATCH --mem=10GB
#SBATCH --cpus-per-task=8
#SBATCH --signal=B:SIGUSR1@30
#SBATCH --open-mode=append
#SBATCH -o /shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/outputs/%x-%j.txt

source /home/s1/wonyoungjang/.bashrc
source /home/s1/wonyoungjang/anaconda3/bin/activate
conda activate diff

echo "Generating."
echo "'c': (3, 1, 0, 2)"

python3 /shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/patch_generation_code/patch_generation_1.py -n 0
