#!/bin/bash

source /home/s1/wonyoungjang/.bashrc

echo "Training SRGAN."

export JOB_NAME="test_SRGAN_128"

accelerate launch --config_file /shared/s1/lab06/wonyoung/diffusers/sd3/config/config_single.yaml \
    /shared/s1/lab06/wonyoung/diffusers/sd3/LDM/train_srgan.py \
    --data_dir="/leelabsg/data/20252_unzip" \
    --train_label_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/data/train_small.csv" \
    --valid_label_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/data/valid_small.csv" \
    --output_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM/results/$JOB_NAME" \
    --resume_from_checkpoint="latest" \
    --axis="c" \
    --seed=42 \
    --allow_tf32 \
    --max_grad_norm=1 \
    --mixed_precision="fp16" \
    --dataloader_num_workers=4 \
    --tracker_project_name=$JOB_NAME \
    --lr_size="128,64,64" \
    --hr_size="128,128,128" \
    --learning_rate_ae=1e-5 \
    --learning_rate_disc=1e-5 \
    --scale_lr \
    --gradient_accumulation_steps=4 \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --valid_batch_size=1 \
    --max_train_steps=100 \
    --discriminator_iter_start=10 \
    --checkpointing_steps=20 \
    #--report_to="wandb"