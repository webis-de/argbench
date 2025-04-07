#!/bin/bash -l
#SBATCH --job-name=prmt-all-data
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=25G
#SBATCH --time=24:00:00
#SBATCH --output argbench/output/test-optuna.out
#SBATCH --error argbench/output/test-optuna.err
#SBATCH --gpus=1
module load Miniforge3
conda activate task-specific3

model=$1

export CODE_PATH="$BIGWORK/task-specific-argument-mining-and-generation"
export DATA_PATH="$BIGWORK/task-specific-argument-mining-and-generation-data"

cd "$CODE_PATH"

python -c "import outlines"