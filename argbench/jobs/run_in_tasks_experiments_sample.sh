#!/bin/bash -l
#SBATCH --job-name=in-task-hpo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=10G
#SBATCH --time=48:00:00
#SBATCH --output argbench/output/in-task-hpo-%j.out
#SBATCH --error argbench/output/in-task-hpo-%j.err
#SBATCH --gpus=1
module load Miniforge3
conda activate task-specific

model=$1
dataset=$2
echo $model


export CODE_PATH="$BIGWORK/task-specific-argument-mining-and-generation"
export DATA_PATH="$BIGWORK/task-specific-argument-mining-and-generation-data"

cd "$CODE_PATH"
if [ -z "$dataset" ]; then
  datasets=$(jq '.test[]'  "${CODE_PATH}/argbench/experiment/configs/experiment_splits.json")
else
  datasets=("\"$dataset\"")
fi
for dataset in ${datasets[@]};
do
echo $dataset
python -m  argbench.experiment.run -c "${CODE_PATH}/argbench/experiment/configs/hpo/in_task_hpo_${dataset:1:-1}.json" \
--base_model "$model" --test_dataset_name "${dataset:1:-1}" --debug --sample

done

export TIME="$(sacct --format=Elapsed -j $SLURM_JOB_ID | tail -n 1 | xargs 2>&1)"
export JOB_ARGUMENTS="hpo;${datasets}.5;in-task;${model};\n"
echo "$SLURM_JOB_ID,$SLURM_JOB_NAME,$TIME,$JOB_ARGUMENTS" >> "$CODE_PATH/argbench/jobs/job-accounting.csv"
