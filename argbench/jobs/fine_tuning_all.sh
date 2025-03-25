#!/bin/bash -l
#SBATCH --job-name=fine-tuning-2
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=12
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --output /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/output/fine-tuning-2.out
#SBATCH --error /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/output/fine-tuning-2.err
#SBATCH --gpus=a100:1
module load Miniforge3
conda activate task-specific
cd /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/
dataset=$1

for model in mistral-7b-inst-3 gemma-2-9b-it mistral-small llama-3-8b-instruct deepseek-r1-distill-1.5b qwen-7b ;
do

  python -m  argbench.experiment.run -c /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/experiment/configs/instruction-fine-tuning/fine_tuning.json \
  --leaderboard-path "/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/runs/fine-tuning-$model-results.csv" \
  --base_model "$model" --test_dataset_name "$dataset"
done