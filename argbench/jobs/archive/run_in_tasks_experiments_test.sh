#!/bin/bash -l
#SBATCH --job-name=in-task
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH --output argbench/output/in-task-%j.out
#SBATCH --error argbench/output/in-task-%j.err
#SBATCH --gpus=1
module load Miniforge3
conda activate task-specific

model=$1
dataset=$2
echo $model



export CODE_PATH="$BIGWORK/task-specific-argument-mining-and-generation"
export DATA_PATH="$BIGWORK/task-specific-argument-mining-and-generation-data"

cd "$CODE_PATH"
python -m  argbench.experiment.run -c "${CODE_PATH}/argbench/experiment/configs/instruction-fine-tuning/in_task_${dataset}.json" \
--model "$model"  --debug

export TIME="$(sacct --format=Elapsed -j $SLURM_JOB_ID | tail -n 1 | xargs 2>&1)"
export JOB_ARGUMENTS="test;${datasets}.5;in-task;${model};\n"
echo "$SLURM_JOB_ID,$SLURM_JOB_NAME,$TIME,$JOB_ARGUMENTS" >> "$CODE_PATH/argbench/jobs/job-accounting.csv"
