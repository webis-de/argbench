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

export CODE_PATH="$BIGWORK/task-specific-argument-mining-and-generation"
export DATA_PATH="$BIGWORK/task-specific-argument-mining-and-generation-data"

python -m  argbench.experiment.run -c "${CODE_PATH}/argbench/experiment/configs/prompting/prompting.json" \
--leaderboard-path "${DATA_PATH}/runs/prompting-$model-results.csv" \
--base_model "$model"
export TIME="$(sacct --format=Elapsed -j $SLURM_JOB_ID | tail -n 1 | xargs 2>&1)"
export JOB_ARGUMENTS="prompting;all-datasets;${model};\n"
echo "$SLURM_JOB_ID,$SLURM_JOB_NAME,$TIME,$JOB_ARGUMENTS" >> "$CODE_PATH/jobs/job-accounting.csv"
