#!/bin/bash -l
#SBATCH --job-name=in-task-hpo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=20G
#SBATCH --time=96:00:00
#SBATCH --gpus=a100:1
module load Miniforge3
conda activate task-specific

model=$1
echo $model


export CODE_PATH="$BIGWORK/task-specific-argument-mining-and-generation"
export DATA_PATH="$BIGWORK/task-specific-argument-mining-and-generation-data"

cd "$CODE_PATH"

datasets=$(jq '.validation[]'  "${CODE_PATH}/argbench/experiment/configs/experiment_splits.json")
for dataset in ${datasets[@]};
do
echo $dataset
python -m  argbench.experiment.run -c "${CODE_PATH}/argbench/experiment/configs/hpo/in_task_hpo.json" \
--base_model "$model" --test_dataset_name "${dataset:1:-1}" --debug

done

export TIME="$(sacct --format=Elapsed -j $SLURM_JOB_ID | tail -n 1 | xargs 2>&1)"
export JOB_ARGUMENTS="hpo;all-datasets;in-task;${model};\n"
echo "$SLURM_JOB_ID,$SLURM_JOB_NAME,$TIME,$JOB_ARGUMENTS" >> "$CODE_PATH/argbench/jobs/job-accounting.csv"