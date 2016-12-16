#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -t 70:00:00
module load cuda
module load cudnn
module load python/3.4.1
python train_model.py