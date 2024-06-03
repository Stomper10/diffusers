#!/bin/bash

#SBATCH --job-name=CXR_unet_nf
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=P2
#SBATCH --nodelist=b14
#SBATCH --time=0-12:00:00
#SBATCH --mem=125GB
#SBATCH --cpus-per-task=32
#SBATCH -o /shared/s1/lab06/wonyoung/diffusers/shell/CXR_unet_nf.txt

source /home/s1/wonyoungjang/.bashrc

echo "Text-to-image Unet finetuning for CheXpert-small data."
echo "Unet finetuning with No Finding images and <lung-xray> text pair."

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# "/shared/s1/lab06/wonyoung/diffusers/CXR_ti_nf"
export DATA_DIR="/shared/s1/lab06/wonyoung/diffusers/data/train_nf.csv"

accelerate launch --config_file /shared/s1/lab06/wonyoung/diffusers/config/config_P2_ds.yaml \
    /shared/s1/lab06/wonyoung/diffusers/scripts/CXR/CXR_train_text_to_image.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --train_data_dir=$DATA_DIR \
    --use_ema \
    --resolution=512 \
    --train_batch_size=32 \
    --gradient_checkpointing \
    --max_train_steps=60000 \
    --learning_rate=5e-05 --scale_lr \
    --max_grad_norm=1 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --output_dir="CXR_unet_nf" \
    --push_to_hub \
    --report_to="wandb" \
    --validation_prompts="A photo of a lung-xray." \
    --checkpointing_steps=500 \
    # --resume_from_checkpoint="latest"
