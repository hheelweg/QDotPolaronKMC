#!/bin/sh
#SBATCH --partition=gpu                 # GPU environment
#SBATCH --nodelist=gpu001               # name of GPU cluster node
#SBATCH --gres=gpu:1                    # nomber of GPUs (can be 2 max for this cluster)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8               # set workers for parallel execution 
#SBATCH --job-name=sample_name          # name of job
#SBATCH --output=sample_name.log        # name of output file

# parameters for parallel execution
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# job submission
python /path/to/python/script/main.py