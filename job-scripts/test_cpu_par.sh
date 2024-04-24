#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=cpu-par-out.%j
#SBATCH --error=cpu-par-err.%j
#SBATCH --time=05:00:00
#SBATCH --account=training2406
#SBATCH --reservation=gpuhack24-2024-04-24
#SBATCH --partition=dc-gpu

source $HOME/miniconda3/bin/activate open-hackathon

python $HOME/newton-evaluation-illustration/timeit-scripts/timeit_cpu_par.py
