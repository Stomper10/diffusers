#!/bin/bash

#SBATCH --job-name=E8_pLDM_UNET3D_whole
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=P2
#SBATCH --time=0-12:00:00
#SBATCH --mem=50GB
#SBATCH --cpus-per-task=8
#SBATCH --signal=B:SIGUSR1@30
#SBATCH --open-mode=append
#SBATCH -o /shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/outputs/%x-%j.txt

source /home/s1/wonyoungjang/.bashrc
source /home/s1/wonyoungjang/anaconda3/bin/activate
conda activate diff

max_restarts=20
scontext=$(scontrol show job ${SLURM_JOB_ID})
restarts=$(echo ${scontext} | grep -o 'Restarts=[0-9]*****' | cut -d= -f2)

function resubmit()
{
    if [[ $restarts -lt $max_restarts ]]; then
        scontrol requeue ${SLURM_JOB_ID}
        exit 0
    else
        echo "Your job is over the Maximum restarts limit"
        exit 1
    fi
}

trap 'resubmit' SIGUSR1

{
echo "Training pLDM (2+1)D PatchUNET from scratch."
echo "'c': (3, 1, 0, 2)"

export JOB_NAME=$SLURM_JOB_NAME
export VAE_PATH="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/results/E1_pLDM_VQGAN3D/checkpoint-770000"

accelerate launch --config_file /shared/s1/lab06/wonyoung/diffusers/sd3/config/config_single.yaml \
    /shared/s1/lab06/wonyoung/diffusers/sd3/LDM_p/train_unet3d_whole.py \
    --pretrained_vae_path=$VAE_PATH \
    --data_dir="/shared/s1/lab06/20252_individual_samples" \
    --train_label_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/data/ukbb_cn_train.csv" \
    --valid_label_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/data/ukbb_cn_valid.csv" \
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
    --max_train_steps=1000000 \
    --checkpointing_steps=10000 \
    --num_samples=3 \
    --use_ema \
    --loss_type="l1" \
    --num_timesteps=1000 \
    --report_to="wandb" \
    #--input_perturbation=0.1 \
    #--gradient_checkpointing \
    # --tiling \
    # --slicing \
    # --use_8bit_adam \
    # --push_to_hub \
} &
wait
exit 0
