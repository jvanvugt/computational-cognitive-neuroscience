#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -t 100:00:00
module load cuda
module load cudnn
module load python/2.7.9
THEANO_FLAGS='mode=FAST_RUN,device=gpu,floatX=float32,lib.cnmem=1' srun python train_model.py