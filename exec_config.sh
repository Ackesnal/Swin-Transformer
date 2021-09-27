#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=Ackesnal
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:tesla-smx2:2
#SBATCH --mem-per-cpu=10G
#SBATCH -o tiny_multi_attn_4unit_29M_50epoch_out.txt
#SBATCH -e tiny_multi_attn_4unit_29M_50epoch_err.txt

module load cuda/10.1.243
module load gnu7/7.3.0
module load mvapich2

srun python -m torch.distributed.launch --nproc_per_node 2 main.py --cfg configs/swin_test_29M.yaml --data-path ../BossNAS/data/imagenet/ --batch-size 256 --multi-attn true --output ./output/tiny_multi_attn_4unit_29M_50epoch
