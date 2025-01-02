#!/bin/bash

source /home/s1/wonyoungjang/.bashrc

echo "Training pLDM (2+1)D PatchUNET from scratch."
echo "'c': (3, 1, 0, 2)"

export JOB_NAME="test_pLDM_UNET3D"
export VAE_PATH="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/E1_pLDM_VQGAN3D/checkpoint-770000"

accelerate launch --config_file /shared/s1/lab06/wonyoung/diffusers/sd3/config/config_single.yaml \
    /shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/train_unet3d_naive.py \
    --pretrained_vae_path=$VAE_PATH \
    --data_dir="/shared/s1/lab06/20252_individual_samples" \
    --train_label_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/data/ukbb_cn_train_small.csv" \
    --valid_label_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/data/ukbb_cn_valid_small.csv" \
    --output_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/$JOB_NAME" \
    --resume_from_checkpoint="latest" \
    --axis="c" \
    --dim_mults="4,8,16,32" \
    --attn_heads=16 \
    --seed=42 \
    --allow_tf32 \
    --max_grad_norm=1 \
    --mixed_precision="fp16" \
    --dataloader_num_workers=4 \
    --tracker_project_name=$JOB_NAME \
    --resolution="76,64,64" \
    --learning_rate=1e-5 \
    --scale_lr \
    --lr_scheduler="polynomial" \
    --gradient_accumulation_steps=1 \
    --train_batch_size=4 \
    --valid_batch_size=4 \
    --max_train_steps=100 \
    --checkpointing_steps=20 \
    --num_samples=1 \
    --use_ema \
    --loss_type="l1" \
    --num_timesteps=1000 \
    #--report_to="wandb" \
    #--input_perturbation=0.1 \
    #--gradient_checkpointing \
    # --tiling \
    # --slicing \
    # --use_8bit_adam \
    # --push_to_hub \
