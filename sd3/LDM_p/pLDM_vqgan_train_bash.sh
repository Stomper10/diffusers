#!/bin/bash

source /home/s1/wonyoungjang/.bashrc

echo "Training pLDM 3D VQGAN from scratch."

export JOB_NAME="test_pLDM_VQGAN3D"

accelerate launch --config_file /shared/s1/lab06/wonyoung/diffusers/sd3/config/config_single.yaml \
    /shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/train_vqgan3d.py \
    --data_dir="/shared/s1/lab06/20252_individual_samples" \
    --train_label_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/data/ukbb_cn_train_small.csv" \
    --valid_label_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/data/ukbb_cn_valid_small.csv" \
    --output_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/$JOB_NAME" \
    --resume_from_checkpoint="latest" \
    --axis="c" \
    --seed=42 \
    --allow_tf32 \
    --max_grad_norm=1 \
    --mixed_precision="fp16" \
    --dataloader_num_workers=4 \
    --tracker_project_name=$JOB_NAME \
    --resolution="76,64,64" \
    --learning_rate_ae=1e-5 \
    --learning_rate_disc=1e-5 \
    --scale_lr \
    --lr_scheduler="constant" \
    --gradient_accumulation_steps=1 \
    --gradient_checkpointing \
    --train_batch_size=2 \
    --valid_batch_size=2 \
    --max_train_steps=100 \
    --discriminator_iter_start=10 \
    --checkpointing_steps=20 \
    --report_to="wandb" \
    # --tiling \
    # --slicing \
    # --use_8bit_adam \
    # --tile_sample_size=64 \
    # --push_to_hub \
    # to write
    # train_label_dir / valid_label_dir / train_batch_size / 
    # valid_batch_size / num_train_epochs / max_train_steps / 
    # checkpointing_steps / validation_epochs
