#!/bin/bash

#SBATCH --job-name=M0_AGE_SYN
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=P2
#SBATCH --time=0-12:00:00
#SBATCH --mem=50GB
#SBATCH --cpus-per-task=8
#SBATCH --signal=B:SIGUSR1@30
#SBATCH --open-mode=append
#SBATCH -o /shared/s1/lab06/wonyoung/diffusers/sd3/LDM/outputs/%x-%j.txt

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
echo "Training CLSSYN from scratch."
echo "'c': (3, 1, 0, 2)"

export JOB_NAME=$SLURM_JOB_NAME

accelerate launch --config_file /shared/s1/lab06/wonyoung/diffusers/sd3/config/config_single.yaml \
    /shared/s1/lab06/wonyoung/diffusers/sd3/LDM/train_cls_syn.py \
    --data_dir="/leelabsg/data/20252_unzip" \
    --train_label_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/data/train_cls.csv" \
    --valid_label_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/data/valid_cls.csv" \
    --gen_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM/gen_volumes/stage2" \
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
    --max_train_steps=1000000 \
    --checkpointing_steps=10000 \
    --n_warmup_steps=2000 \
    --report_to="wandb" \
    --task="age"
    #--traditional_transform
} &
wait
exit 0
