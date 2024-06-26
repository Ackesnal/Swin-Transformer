#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=Ackesnal
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:tesla-smx2:4
#SBATCH --mem-per-cpu=10G
#SBATCH -o tiny_original_50epoch_out.txt
#SBATCH -e err.txt

source activate swin
module load cuda/10.1.243
module load gnu7/7.3.0
module load mvapich2

srun python -m torch.distributed.launch --nproc_per_node 4 --master_port 10162 main.py --cfg configs/swin_test_tiny.yaml --data-path ../BossNAS/data/imagenet/ --batch-size 256 --output ./output/original
