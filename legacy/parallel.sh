#!/bin/bash
#SBATCH --partition=normal                  # set queue
#SBATCH --job-name=exciton_diffusion            	    # Job name
#SBATCH --mail-type=END,FAIL                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --output=parallel_%j.log            # Standard output and error log
#SBATCH --nodes=1                          # Run all processes on a single node
#SBATCH --ntasks=1                          # Run a single task
#SBATCH --cpus-per-task=48                  # Number of CPU cores per task
#SBATCH --time=500:00:00                    # Time limit hrs:min:sec
# #SBATCH --mem=1gb                           # Job memory request
srun /home/tsheehan/.conda/envs/pythenv/bin/python3 main_bashversion.py
