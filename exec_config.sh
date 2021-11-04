#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=Ackesnal
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=10G
#SBATCH -o original_swin_28M_4.5G_300epoch_out.txt
#SBATCH -e original_swin_28M_4.5G_300epoch_err.txt

module load cuda/10.1.243
module load gnu7/7.3.0
module load mvapich2

srun python -m torch.distributed.launch --nproc_per_node=2 main.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path ../BossNAS/data/imagenet/ --batch-size 512 --output ./output/original_swin_28M_4.5G_300epoch --multi-attn false --use-checkpoint
