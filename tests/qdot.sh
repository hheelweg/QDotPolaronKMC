#!/bin/sh
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8        
#SBATCH --job-name=qdot
#SBATCH --output=test.log

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Job submission
python -c "import qdotkmc,inspect; print('Using:', qdotkmc.__file__)"
time python /home/hheelweg/QDots/QDotPolaronKMC/tests/box_main.py

# USAGE : need to adjust python path and file name for execution