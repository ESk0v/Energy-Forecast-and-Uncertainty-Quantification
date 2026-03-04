#!/bin/bash

#SBATCH --job-name=Abvaerk
#SBATCH --output=Abvaerk.out
#SBATCH --error=Abvaerk.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=15
#SBATCH --mem=24G
#SBATCH --gres=gpu:1            # request 1 GPU
#SBATCH --partition=l4          # correct GPU partition

# ==========================================
#        ABVAERK LSTM PIPELINE RUNNER
# ==========================================
#
# Usage examples:
#
#   sbatch run.sh 
#
# Arguments:
#   mode        | --mode full       | (tune | train | ensemble | full)  | (Allways relevant)
#   n_trials    | --n_trials 1      | (Any Number )                     | (Only relevant for tuning)
#
# ==========================================

# Activate virtual environment
cd /ceph/project/SW6-Group18-Abvaerk
source /ceph/project/SW6-Group18-Abvaerk/.venv/bin/activate
python -u NewModelFolder/Main.py --mode full --n_trials 50 | tee -a Abvaerk_live.out