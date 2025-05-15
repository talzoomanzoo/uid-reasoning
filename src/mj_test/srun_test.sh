#!/bin/bash

# Load necessary modules
module load python/3.10.16
module load cuda/12.5

# Activate your virtual environment
source /home/hojinkim/miniconda3/envs/reasoning/bin/activate

# Run your Python script using srun
srun --ntasks=1 --cpus-per-task=4 --mem=16G python /home/hojinkim/mjgwak/search-o1-dev/scripts/mj_test/mj_two_stage_sampling.py