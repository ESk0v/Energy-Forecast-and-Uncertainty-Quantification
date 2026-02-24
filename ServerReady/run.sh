#!/bin/bash

#SBATCH --job-name=Abvaerk
#SBATCH --output=Abvaerk.out
#SBATCH --error=Abvaerk.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=15
#SBATCH --mem=24G
#SBATCH --gres=gpu:1        # request 1 GPU
#SBATCH --partition=l4       # correct GPU partition

# Load modules
module load python/3.10
module load cuda             # ensures CUDA toolkit & drivers are available

# Activate virtual environment
source ~/ceph/projects/SW6-Group18-Abvaerk/.venv/bin/activate

# Move to project folder
cd ~/ceph/projects/SW6-Group18-Abvaerk

# Run the tuning script
python ./ServerReady/ModelTuning/LSTMTraining.py