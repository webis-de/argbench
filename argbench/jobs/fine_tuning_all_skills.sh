#!/bin/bash -l
#SBATCH --job-name=fine-tuning-2
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=12
#SBATCH --mem=20G
#SBATCH --time=96:00:00
#SBATCH --output argbench/output/fnt-skills.out
#SBATCH --error argbench/output/fnt-skills.err
#SBATCH --gpus=a100:1
module load Miniforge3
conda activate task-specific

model=$1
dataset=$2
echo $model
echo $dataset

export CODE_PATH="$BIGWORK/task-specific-argument-mining-and-generation"
export DATA_PATH="$BIGWORK/task-specific-argument-mining-and-generation-data"

cd "$CODE_PATH"


for skill in mining perspective-assessment quality assessment generation reasoning ;
do

python -m  argbench.experiment.run -c "${CODE_PATH}/argbench/experiment/configs/instruction-fine-tuning/fine_tuning.json" \
--leaderboard-path "${DATA_PATH}/runs/fine-tuning-$model-skills-results.csv" \
--base_model "$model" --test_dataset_name "$dataset" --skill-filter "${skill}" --debug
python -c "print(1*2)"

export TIME="$(sacct --format=Elapsed -j $SLURM_JOB_ID | tail -n 1 | xargs 2>&1)"
export JOB_ARGUMENTS="fine-tuning-skills;${dataset};${model};\n"
echo "$SLURM_JOB_ID,$SLURM_JOB_NAME,$TIME,$JOB_ARGUMENTS" >> "$CODE_PATH/jobs/job-accounting.csv"

done