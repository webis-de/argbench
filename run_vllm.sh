#!/bin/bash -l
#SBATCH --job-name=prmt-all-data
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=25G
#SBATCH --time=96:00:00
#SBATCH --output argbench/output/prmt-all-data.out
#SBATCH --error argbench/output/prmt-all-data.err
#SBATCH --gpus=a100:1
module load Miniforge3
conda activate task-specific

export CODE_PATH="$BIGWORK/task-specific-argument-mining-and-generation"

python "$CODE_PATH/run_vllm_locally.py"