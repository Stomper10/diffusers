#!/bin/bash

#SBATCH --job-name=E1_wLDM_VQGAN3D
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=P2
#SBATCH --time=0-12:00:00
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=8
#SBATCH --signal=B:SIGUSR1@30
#SBATCH --open-mode=append
#SBATCH -o /shared/s1/lab06/wonyoung/diffusers/sd3/LDM_w/outputs/%x-%j.txt

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
echo "Training wLDM 3D VQGAN from scratch."
echo "'c': (3, 1, 0, 2)"

export JOB_NAME=$SLURM_JOB_NAME

accelerate launch --config_file /shared/s1/lab06/wonyoung/diffusers/sd3/config/config_single.yaml \
    /shared/s1/lab06/wonyoung/diffusers/sd3/LDM_w/train_vqgan3d.py \
    --data_dir="/shared/s1/lab06/20252_individual_samples" \
    --train_label_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/data/ukbb_cn_train.csv" \
    --valid_label_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/data/ukbb_cn_valid.csv" \
    --output_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM_w/results/$JOB_NAME" \
    --resume_from_checkpoint="latest" \
    --axis="c" \
    --seed=21 \
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
    --max_train_steps=2000000 \
    --discriminator_iter_start=10000 \
    --checkpointing_steps=10000 \
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
} &
wait
exit 0
