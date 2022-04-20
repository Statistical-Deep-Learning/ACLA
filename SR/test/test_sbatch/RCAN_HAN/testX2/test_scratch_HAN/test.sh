#!/bin/bash

#SBATCH --get-user-env
#SBATCH -J super-test
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=20000
#SBATCH -p gpu
#SBATCH -q wildfire

#SBATCH --gres=gpu:1

#SBATCH -t 0-1:00
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err


eval "$(conda shell.bash hook)"
conda activate super
module purge
module load cuda/10.2.89
module load cudnn/8.1.0

cd /home/ywan1053/UNSR-train/test/

python main.py \
--template RCAN_HAN_x2_scratch_gammaL1C1_fixed \
--testset Set5

python main.py \
--template RCAN_HAN_x2_scratch_gammaL1C1_fixed \
--testset Set14

python main.py \
--template RCAN_HAN_x2_scratch_gammaL1C1_fixed \
--testset B100

python main.py \
--template RCAN_HAN_x2_scratch_gammaL1C1_fixed \
--testset Urban100

python main.py \
--template RCAN_HAN_x2_scratch_gammaL1C1_fixed \
--testset Manga109