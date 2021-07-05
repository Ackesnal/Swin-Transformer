#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=Ackesnal
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:tesla-smx2:4
#SBATCH --mem-per-cpu=20G
#SBATCH -o out.txt
#SBATCH -e err.txt

module load anaconda/3.6
source activate swin
module load cuda/10.1.234
module load gnu7/7.3.0
module load mvapich2

srun python -m torch.distributed.launch --nproc_per_node 4 --master_port 10062 main.py --cfg configs/swin_test_tiny.yaml --data-path ../BossNAS/data/imagenet/ --batch-size 1024 --window-size=[2,4,8,16,32]
