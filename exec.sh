#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=Ackesnal
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:tesla-smx2:2
#SBATCH --mem-per-cpu=10G
#SBATCH -o swin_tiny_shufflev2_300epoch_ratio0.5_out.txt
#SBATCH -e swin_tiny_shufflev2_300epoch_ratio0.5_err.txt

srun python -m torch.distributed.launch --nproc_per_node 2 --master_port 10192 main.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path ../data/imagenet/ --batch-size 512 --use-checkpoint --output ./output/swin_tiny_shufflev2_300epoch_ratio0.5
