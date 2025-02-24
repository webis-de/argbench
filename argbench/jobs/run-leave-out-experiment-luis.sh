#!/bin/bash -l
#SBATCH --job-name=task-specific
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/src/output/output.out
#SBATCH --error /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/src/output/error.err
#SBATCH --gpus=1
module load Miniconda3
conda activate task-specific

python /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/src/experiment/run.py
