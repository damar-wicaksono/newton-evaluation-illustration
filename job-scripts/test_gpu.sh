#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=gpu-out.%j
#SBATCH --error=gpu-err.%j
#SBATCH --time=02:00:00
#SBATCH --account=training2406
#SBATCH --reservation=gpuhack24-2024-04-24
#SBATCH --partition=dc-gpu
#SBATCH --gres=gpu:1

source $HOME/miniconda3/bin/activate open-hackathon

python $HOME/newton-evaluation-illustration/timeit-scripts/timeit_gpu.py
