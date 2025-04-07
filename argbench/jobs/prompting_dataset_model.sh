#!/bin/bash -l
#SBATCH --job-name=prmt-dataset
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=12
#SBATCH --mem=25G
#SBATCH --time=24:00:00
#SBATCH --output argbench/output/prmt-dataset.out
#SBATCH --error argbench/output/prmt-dataset.err
#SBATCH --gpus=a100:1


module load Miniforge3
conda activate task-specific

model=$1
dataset=$2

export CODE_PATH="$BIGWORK/task-specific-argument-mining-and-generation"
export DATA_PATH="$BIGWORK/task-specific-argument-mining-and-generation-data"

cd "$CODE_PATH"

python -m  argbench.experiment.run -c "${CODE_PATH}/argbench/experiment/configs/prompting/prompting.json" \
--leaderboard-path "${DATA_PATH}/runs/prompting-$model-results.csv" --base_model "$model" --test_dataset_name "$dataset"

export TIME="$(sacct --format=Elapsed -j $SLURM_JOB_ID | tail -n 1 | xargs 2>&1)"
export JOB_ARGUMENTS="prompting;${dataset};${model};\n"
echo "$SLURM_JOB_ID,$SLURM_JOB_NAME,$TIME,$JOB_ARGUMENTS" >> "$CODE_PATH/argbench/jobs/job-accounting.csv"