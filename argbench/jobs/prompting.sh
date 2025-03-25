#!/bin/bash -l
#SBATCH --job-name=prompting
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=12
#SBATCH --mem=25G
#SBATCH --time=24:00:00
#SBATCH --output /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/output/prompting.out
#SBATCH --error /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/output/prompting.err
#SBATCH --gpus=a100:1
module load Miniforge3
conda activate task-specific
cd /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/
model=$1

python -m  argbench.experiment.run -c /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/experiment/configs/prompting/prompting.json \
--leaderboard-path "/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/runs/prompting-$model-results.csv" \
--base_model "$model"
