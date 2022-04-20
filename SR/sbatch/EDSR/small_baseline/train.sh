#!/bin/bash

#SBATCH --get-user-env
#SBATCH -J super
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=24000
#SBATCH -p gpu
#SBATCH -q wildfire

#SBATCH --gres=gpu:V100:1

#SBATCH -t 0-168:00
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err


eval "$(conda shell.bash hook)"
conda activate super
module purge
module load cuda/10.2.89
module load cudnn/8.1.0

cd /home/ywan1053/UNSR-train/
python main.py \
--model EDSR \
--save EDSR_small_test \
--scale 2 \
--n_GPUs 1 \
--n_resblocks 16 \
--n_feats 128 \
--n_threads 4 \
--chop \
--save_results \
--print_every 100 \
--batch_size 16 \
--test_every 1000 \
--ext bin \
--rgb_range 1 \
--res_scale 0.1 \
--lr 1e-4 \
--lr_decay 200 \
--gamma 0.5 \
--epochs 1000 \
--loss 1*L1 \
--print_model \
--dir_data /data/yyang409/yancheng/data/SR_data \
--result_path /data/yyang409/yancheng/SR_results_torch16/EDSR_small_baseline_NLSA/ \
--patch_size 96
