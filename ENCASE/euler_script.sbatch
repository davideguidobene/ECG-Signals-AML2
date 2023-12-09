#!/usr/bin/bash
#SBATCH --ntasks-per-node=4
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --job-name=aml2
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --gpus=rtx_3090:1

module load gcc/8.2.0 python_gpu/3.11.2 hdf5/1.10.1 eth_proxy

pip3 install xgboost --user

CUDA_VISIBLE_DEVICES=0 python3.11 classifier.py
