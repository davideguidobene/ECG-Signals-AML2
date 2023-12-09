#!/usr/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=96:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name=parallel
#SBATCH --output=parallel.out
#SBATCH --error=parallel.err
#SBATCH --gpus=rtx_3090:1

module load gcc/8.2.0 python_gpu/3.11.2 hdf5/1.10.1 eth_proxy

CUDA_VISIBLE_DEVICES=0 python3.11 centerwave_parallel.py


