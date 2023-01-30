#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=0.5idleswin
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2G
#SBATCH -o idle_swin_r0.5_new_out.txt
#SBATCH -e idle_swin_r0.5_new_err.txt

python -m torch.distributed.launch --nproc_per_node 2 --master_port 10121 main.py --cfg configs/swin/swin_tiny_patch4_window7_224.yaml --data-path ../data/imagenet/ --batch-size 256 --output ./output/idle_swin_r0.5_new --pretrained ./swin_tiny_patch4_window7_224.pth --use-checkpoint --accumulation-steps 2