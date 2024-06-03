#!/bin/bash

#SBATCH --job-name=VAE_train_test2d
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=P2
#SBATCH --nodelist=b14
#SBATCH --time=0-12:00:00
#SBATCH --mem=125GB
#SBATCH --cpus-per-task=32
#SBATCH -o /shared/s1/lab06/wonyoung/diffusers/examples/_brain_ldm/script/%x.txt

source /home/s1/wonyoungjang/.bashrc

echo "Text-to-image Unet finetuning for CheXpert-small data."

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="/shared/s1/lab06/20252_individual_samples"

accelerate launch --config_file /shared/s1/lab06/wonyoung/diffusers/examples/_brain_ldm/config/config_P2.yaml \
    /shared/s1/lab06/wonyoung/diffusers/examples/_brain_ldm/train_vae.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --data_dir=$DATA_DIR \
    --train_label_dir="/shared/s1/lab06/wonyoung/diffusers/examples/_brain_ldm/data/ukbb_cn_train.csv" \
    --valid_label_dir="/shared/s1/lab06/wonyoung/diffusers/examples/_brain_ldm/data/ukbb_cn_valid.csv" \
    --output_dir="/shared/s1/lab06/wonyoung/diffusers/examples/_brain_ldm/results/${SLURM_JOB_NAME}" \
    --seed=42 \
    --resolution="160,224,160" \
    --train_batch_size=16 \
    --num_train_epochs=100 \
    --max_train_steps=10 \
    --gradient_accumulation_steps=1 \
    --gradient_checkpointing \
    --learning_rate=1e-4 \
    --scale_lr \
    --use_ema \
    --max_grad_norm=1 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=500 \
    --dataloader_num_workers=8 \
    --mixed_precision="fp16" \
    --report_to="wandb" \
    --checkpointing_steps=1 \
    --validation_epochs=1 \
    --tracker_project_name=$SLURM_JOB_NAME \
    # --resume_from_checkpoint="latest"
    # --use_8bit_adam
    # --allow_tf32
    # --push_to_hub \
