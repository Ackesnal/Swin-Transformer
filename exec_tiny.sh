#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=Ackesnal
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:tesla-smx2:4
#SBATCH --mem-per-cpu=10G
#SBATCH -o normed_shuffle_swin_7M_1G_300epoch_out.txt
#SBATCH -e normed_shuffle_swin_7M_1G_300epoch_err.txt

module load cuda/10.1.243
module load gnu7/7.3.0
module load mvapich2

srun python -m torch.distributed.launch --nproc_per_node=4 main.py --cfg configs/swin_test_tiny_7M_shuffle.yaml --data-path ../data/imagenet/ --batch-size 1024 --multi-attn true --output ./output/normed_shuffle_swin_7M_1G_300epoch --use-checkpoint
