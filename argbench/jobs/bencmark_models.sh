#!/bin/bash -l
#SBATCH --job-name=benchmarking_models
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=6
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output argbench/output/benchmarking_models.out
#SBATCH --error argbench/output/benchmarking_models.err
#SBATCH --gpus=a100:2

export CODE_PATH="$BIGWORK/task-specific-argument-mining-and-generation"

module load Miniforge3
conda activate task-specific
python "$CODE_PATH/benchmark_performance.py"
