#!/bin/bash

source /home/s1/wonyoungjang/.bashrc

echo "Training (2+1)D UNET from scratch."

export JOB_NAME="test_UNET3D_128"
export VAE_PATH="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM/results/E0_VQGAN3D_128/checkpoint-300000"

accelerate launch --config_file /shared/s1/lab06/wonyoung/diffusers/sd3/config/config_single.yaml \
    /shared/s1/lab06/wonyoung/diffusers/sd3/LDM/train_unet3d_test.py \
    --pretrained_vae_path=$VAE_PATH \
    --data_dir="/leelabsg/data/20252_unzip" \
    --train_label_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/data/train_small.csv" \
    --valid_label_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/data/valid_small.csv" \
    --output_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM/results/$JOB_NAME" \
    --resume_from_checkpoint="latest" \
    --axis="c" \
    --dim_mults="16,32,64,128" \
    --attn_heads=24 \
    --seed=42 \
    --allow_tf32 \
    --max_grad_norm=1 \
    --mixed_precision="fp16" \
    --dataloader_num_workers=4 \
    --tracker_project_name=$JOB_NAME \
    --resolution="128,32,32" \
    --learning_rate=1e-5 \
    --scale_lr \
    --lr_scheduler="polynomial" \
    --gradient_accumulation_steps=1 \
    --train_batch_size=4 \
    --valid_batch_size=4 \
    --max_train_steps=100 \
    --checkpointing_steps=20 \
    --num_samples=1 \
    --loss_type="l2" \
    --num_timesteps=1000 \
    --input_perturbation=0.1 \
    --noise_offset=0.1 \
    --snr_gamma=5.0 \
    # --report_to="wandb" \