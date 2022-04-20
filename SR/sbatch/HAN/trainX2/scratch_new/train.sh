#!/bin/bash

#SBATCH --get-user-env
#SBATCH -J super
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --mem=24000
#SBATCH -p yyanggpu1
#SBATCH -q yyanggpu1

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
--model NEW_HAN \
--save HAN_scratch_gammaL0C0 \
--scale 2 \
--n_GPUs 1 \
--n_resgroups 10 \
--n_resblocks 20 \
--n_feats 64 \
--n_threads 1 \
--reset \
--chop \
--save_results \
--print_every 100 \
--batch_size 16 \
--test_every 1000 \
--ext bin \
--lr 1e-4 \
--lr_decay 200 \
--epochs 1000 \
--loss 1*L1 \
--print_model \
--dir_data /data/yyang409/yancheng/data/SR \
--result_path /data/yyang409/yancheng/SR_results_torch16/NEW_HAN_x2/ \
--patch_size 96 \
--initial_gamma_LAM 0 \
--initial_gamma_CSAM 0

