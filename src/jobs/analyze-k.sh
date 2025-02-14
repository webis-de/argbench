#!/bin/bash -l
#SBATCH --job-name=task-specific
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --time=24:00:00
#SBATCH --output /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/src/output/hpo-stance-barhaim.out
#SBATCH --error /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/src/output/hpo-stance-barhaim.err
#SBATCH --gpus=a100:1
module load Miniforge3
conda activate task-specific
for i in {1..5}
do
  k = $((i ** 2))
  python /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/src/experiment/run.py
  -c /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/src/experiment/configs/prompting/barhaim17_mistral_prompting.json
  --train_subsample_amount "$k" --leaderboard-path
done
