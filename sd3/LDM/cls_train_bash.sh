#!/bin/bash

source /home/s1/wonyoungjang/.bashrc

echo "Training CLS from scratch."

export JOB_NAME="test_CLS_128"

accelerate launch --config_file /shared/s1/lab06/wonyoung/diffusers/sd3/config/config_single.yaml \
    /shared/s1/lab06/wonyoung/diffusers/sd3/LDM/train_cls.py \
    --data_dir="/leelabsg/data/20252_unzip" \
    --train_label_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/data/train_cls_small.csv" \
    --valid_label_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/data/valid_cls_small.csv" \
    --output_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM/results/$JOB_NAME" \
    --resume_from_checkpoint="latest" \
    --axis="c" \
    --seed=42 \
    --allow_tf32 \
    --max_grad_norm=1 \
    --mixed_precision="fp16" \
    --dataloader_num_workers=4 \
    --tracker_project_name=$JOB_NAME \
    --resolution="128,128,128" \
    --learning_rate=1e-5 \
    --scale_lr \
    --lr_scheduler="cosine_with_restarts" \
    --gradient_accumulation_steps=1 \
    --train_batch_size=16 \
    --valid_batch_size=4 \
    --max_train_steps=100 \
    --checkpointing_steps=20 \
    --task="age" \
    --traditional_transform
    #--report_to="wandb" 