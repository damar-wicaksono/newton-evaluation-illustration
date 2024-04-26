#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=gpu-splits-rev-out.%j
#SBATCH --error=gpu-splits-rev-err.%j
#SBATCH --time=02:00:00
#SBATCH --account=training2406
##SBATCH --reservation=gpuhack24-2024-04-25
#SBATCH --partition=dc-gpu
#SBATCH --gres=gpu:1

source /p/home/jusers/wicaksono1/jureca/miniconda3/bin/activate open-hackathon

python /p/home/jusers/wicaksono1/jureca/newton-evaluation-illustration/timeit-scripts/timeit_gpu_splits_rev.py $DEGREE $SPLITS
