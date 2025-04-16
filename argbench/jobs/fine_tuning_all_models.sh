#!/bin/bash -l
#SBATCH --job-name=ft-all-models
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=12
#SBATCH --mem=20G
#SBATCH --time=96:00:00
#SBATCH --output argbench/output/ft-all-models.out
#SBATCH --error argbench/output/ft-all-models.err
#SBATCH --gpus=a100:1
module load Miniforge3
conda activate task-specific

export CODE_PATH="$BIGWORK/task-specific-argument-mining-and-generation"
export DATA_PATH="$BIGWORK/task-specific-argument-mining-and-generation-data"


cd $CODE_PATH
dataset=$1




for model in mistral-7b-inst-3 gemma-2-9b-it mistral-small llama-3-8b-instruct deepseek-r1-distill-1.5b qwen-7b ;
do

  python -m  argbench.experiment.run -c "${CODE_PATH}/argbench/experiment/configs/instruction-fine-tuning/fine_tuning.json" \
  --leaderboard-path "${DATA_PATH}/runs/fine-tuning-$model-results.csv" \
  --base_model "$model" --test_dataset_name "$dataset"
done


export TIME="$(sacct --format=Elapsed -j $SLURM_JOB_ID | tail -n 1 | xargs 2>&1)"
export JOB_ARGUMENTS="fine-tuning;${dataset};all-models;\n"
echo "$SLURM_JOB_ID,$SLURM_JOB_NAME,$TIME,$JOB_ARGUMENTS" >> "$CODE_PATH/argbench/jobs/job-accounting.csv"