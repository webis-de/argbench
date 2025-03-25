#!/bin/bash -l
#SBATCH --job-name=task-specific
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --time=24:00:00
#SBATCH --output argbench/output/analyze-k.out
#SBATCH --error /argbench/output/analyze-k.err
#SBATCH --gpus=a100:1

export CODE_PATH="$BIGWORK/task-specific-argument-mining-and-generation"
export DATA_PATH="$BIGWORK/task-specific-argument-mining-and-generation-data"

module load Miniforge3
conda activate task-specific
cd "$CODE_PATH"
for i in {1..5}
do
  k=$((i ** 2))
  cd "$CODE_PATH"

  python -m argbench.experiment.run -c "$CODE_PATH/argbench/experiment/configs/prompting/prompting.json" \
  --train_subsample_amount "$k" --leaderboard-path "$DATA_PATH/runs/prompting-llama3-instruct-results-k.csv"
done
