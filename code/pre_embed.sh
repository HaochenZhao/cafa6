#!/bin/bash
#SBATCH --job-name=CAFA-pre
#SBATCH --output=./slurm_logs/pre/CAFA-650M_3B_%j.out
#SBATCH --error=./slurm_logs/pre/CAFA-650M_3B_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100-80:1  
#SBATCH --partition=gpu-long

source ~/.bashrc
conda activate cafa
cd ~/cafa


# 2580
python precompute_embeddings.py \
    --model_name esm2_t36_3B_UR50D \
    --batch_size 4 \
    --pooling mean \
    --max_seq_len 2048
# 5120
python precompute_embeddings.py \
    --model_name esm2_t48_15B_UR50D \
    --batch_size 4 \
    --pooling mean \
    --max_seq_len 2048

