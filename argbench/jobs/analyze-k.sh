#!/bin/bash -l
#SBATCH --job-name=task-specific
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --time=24:00:00
#SBATCH --output /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/output/analyze-k.out
#SBATCH --error /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/output/analyze-k.err
#SBATCH --gpus=a100:1
module load Miniforge3
conda activate task-specific
cd /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/
for i in {1..5}
do
  k=$((i ** 2))
  cd /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/

  python -m argbench.experiment.run -c /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/experiment/configs/prompting/lamma3_instruct_prompting.json \
  --train_subsample_amount "$k" --leaderboard-path /bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/runs/prompting-llama3-instruct-results-k.csv
done
