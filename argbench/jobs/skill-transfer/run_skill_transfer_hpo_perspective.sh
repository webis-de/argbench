#!/bin/bash -l
#SBATCH --job-name=skill-transfer-hpo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output argbench/output/skill-transfer-hpo-%j.out
#SBATCH --error argbench/output/skill-transfer-hpo-%j.err
#SBATCH --gpus=1
module load Miniforge3
conda activate task-specific

model=$1
echo $model


export CODE_PATH="$BIGWORK/task-specific-argument-mining-and-generation"
export DATA_PATH="$BIGWORK/task-specific-argument-mining-and-generation-data"

cd "$CODE_PATH"

python -m  argbench.experiment.run -c "${CODE_PATH}/argbench/experiment/configs/hpo/skill_transfer_hpo.json" \
--base_model "$model"  --debug --sample --test_dataset_name "fallacy_detection_logic_jin22" --skill_filter "perspective-assessment"


export TIME="$(sacct --format=Elapsed -j $SLURM_JOB_ID | tail -n 1 | xargs 2>&1)"
export JOB_ARGUMENTS="hpo;fallacy_detection_logic_jin22;skill-transfer-perspective-assessment;${model};\n"
echo "$SLURM_JOB_ID,$SLURM_JOB_NAME,$TIME,$JOB_ARGUMENTS" >> "$CODE_PATH/argbench/jobs/job-accounting.csv"
