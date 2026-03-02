#!/bin/bash

#SBATCH --job-name=Abvaerk
#SBATCH --output=Abvaerk.out
#SBATCH --error=Abvaerk.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=15
#SBATCH --mem=24G
#SBATCH --gres=gpu:1        # request 1 GPU
#SBATCH --partition=l4       # correct GPU partition

# Activate virtual environment
cd /ceph/project/SW6-Group18-Abvaerk
source /ceph/project/SW6-Group18-Abvaerk/.venv/bin/activate
python NewModelFolder/Main.py --mode full --n_trials 1
