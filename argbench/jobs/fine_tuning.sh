#!/bin/bash -l
#SBATCH --job-name=fine-tuning
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --time=24:00:00
#SBATCH --output /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/output/fine-tuning.out
#SBATCH --error /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/output/fine-tuning.err
#SBATCH --gpus=a100:1
module load Miniforge3
conda activate task-specific
cd /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/
model=$1
dataset=$2
python -m  argbench.experiment.run -c /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/experiment/configs/prompting/fine_tuning.json \
--leaderboard-path "/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/runs/prompting-$model-results.csv" \
--base_model "$model" --test_dataset "$dataset"