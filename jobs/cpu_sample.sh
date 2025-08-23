#!/bin/sh
#SBATCH --nodes=1
#SBATCH --partition=normal              # CPU environment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8               # set workers for parallel execution 
#SBATCH --job-name=sample_name          # name of job
#SBATCH --output=sample_name.log        # name of output file

# parameters for parallel execution
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ensure GPU stays off (run on CPU)
export QDOT_USE_GPU=0

# job submission
python /path/to/python/script/main.py
