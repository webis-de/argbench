#!/bin/bash -l
#SBATCH --job-name=cross-task
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=96:00:00
#SBATCH --output argbench/output/cross-task-%j.out
#SBATCH --error argbench/output/cross-task-%j.err
#SBATCH --gpus=1
module load Miniforge3
conda activate task-specific

model=$1
echo $model
dataset=$2

export CODE_PATH="$BIGWORK/task-specific-argument-mining-and-generation"
export DATA_PATH="$BIGWORK/task-specific-argument-mining-and-generation-data"

cd "$CODE_PATH"


python -m  argbench.experiment.run -c "${CODE_PATH}/argbench/experiment/configs/hpo/cross_task_hpo_${dataset}.json" \
--base_model "$model"  --debug



export TIME="$(sacct --format=Elapsed -j $SLURM_JOB_ID | tail -n 1 | xargs 2>&1)"
export JOB_ARGUMENTS="hpo;${dataset}.5;cross-task;${model};\n"
echo "$SLURM_JOB_ID,$SLURM_JOB_NAME,$TIME,$JOB_ARGUMENTS" >> "$CODE_PATH/argbench/jobs/job-accounting.csv"
