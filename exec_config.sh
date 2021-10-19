#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=Ackesnal
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:tesla-smx2:2
#SBATCH --mem-per-cpu=10G
#SBATCH -o multi_type_3unit_7.3M_shuffle_50epoch_out.txt
#SBATCH -e multi_type_3unit_7.3M_shuffle_50epoch_err.txt

module load cuda/10.1.243
module load gnu7/7.3.0
module load mvapich2

srun python -m torch.distributed.launch --nproc_per_node=2 main.py --cfg configs/swin_test_7M.yaml --data-path ../BossNAS/data/imagenet/ --batch-size 512 --multi-attn true --output ./output/multi_type_3unit_7.3M_shuffle_50epoch --use-checkpoint
