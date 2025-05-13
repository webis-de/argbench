#!/bin/bash -l
#SBATCH --job-name=prmt-all-data
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=10G
#SBATCH --time=12:00:00
#SBATCH --output argbench/output/run_vllm.out
#SBATCH --error argbench/output/run_vllm.err
#SBATCH --gpus=1
module load Miniforge3
conda activate few-shot-priming

export CODE_PATH="$BIGWORK/task-specific-argument-mining-and-generation"

python "$CODE_PATH/run_vllm_locally.py"
