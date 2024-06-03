#!/bin/bash

source /data/home/wonyoungjang/.bashrc

echo "Test training for VAE 2D."

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="/media/leelabsg-storage1/DATA/UKBB/bulk/20252_numpy/20252_individual_samples"
export JOB_NAME="testrun"

accelerate launch --config_file /data/home/wonyoungjang/diffusers/labs/_brain_ldm/config/config_P2.yaml \
    /data/home/wonyoungjang/diffusers/labs/_brain_ldm/train_vae.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --data_dir=$DATA_DIR \
    --train_label_dir="/data/home/wonyoungjang/diffusers/labs/_brain_ldm/data/ukbb_cn_train_small.csv" \
    --valid_label_dir="/data/home/wonyoungjang/diffusers/labs/_brain_ldm/data/ukbb_cn_valid_small.csv" \
    --output_dir="/data/home/wonyoungjang/diffusers/labs/_brain_ldm/result/$JOB_NAME" \
    --seed=42 \
    --resolution="128,128,128" \
    --train_batch_size=1 \
    --valid_batch_size=1 \
    --dataloader_num_workers=4 \
    --num_train_epochs=100 \
    --max_train_steps=10 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-4 \
    --scale_lr \
    --use_ema \
    --max_grad_norm=1 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=500 \
    --checkpointing_steps=5 \
    --validation_epochs=5 \
    --tracker_project_name=$JOB_NAME \
    --mixed_precision="fp16" \
    --variant="fp16" \
    --allow_tf32 \
    --use_8bit_adam \
    --gradient_checkpointing \
    --tiling \
    --tile_sample_size=64 \
    --slicing \
    --report_to="wandb" \
    --resume_from_checkpoint="latest" \
    # --push_to_hub \
    # to write
    # train_label_dir / valid_label_dir / train_batch_size / 
    # valid_batch_size / num_train_epochs / max_train_steps / 
    # checkpointing_steps / validation_epochs