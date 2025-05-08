#!/bin/bash -l
#SBATCH --job-name=prmt-all-data
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=12
#SBATCH --mem=25G
#SBATCH --time=96:00:00
#SBATCH --output argbench/output/prmt-all-data-%j.out
#SBATCH --error argbench/output/prmt-all-data-%j.err
#SBATCH --gpus=a100:1
module load Miniforge3
conda activate task-specific

model=$1

export CODE_PATH="$BIGWORK/task-specific-argument-mining-and-generation"
export DATA_PATH="$BIGWORK/task-specific-argument-mining-and-generation-data"

cd "$CODE_PATH"

if [ -z "$model" ]; then

#models=$(jq '.model_configs[].label' "${CODE_PATH}/argbench/experiment/configs/prompting/prompting.json")
#models=("gemma-9-2b-it" "gemma-2-27b-it" "mistral-7b-inst-3" "mistral-small")
models=("gemma-9-2b-it" "gemma-2-27b-it" "mistral-7b-inst-3" "mistral-small" "deepseek-r1-distill-1.5b" "deepseek-r1-distill-32b")
else

models=($model)
fi

for model in ${models[@]};
do
  echo "$model"
python -m  argbench.experiment.run -c "${CODE_PATH}/argbench/experiment/configs/prompting/prompting.json" \
--leaderboard-path "${DATA_PATH}/runs/prompting-$model-results.csv" --base_model "$model" --debug

python -m  argbench.experiment.run -c "${CODE_PATH}/argbench/experiment/configs/prompting/prompting.json" \
--leaderboard-path "${DATA_PATH}/runs/prompting-$model-results.csv" --base_model "$model" --debug --is_chain_of_thoughts

python -m  argbench.experiment.run -c "${CODE_PATH}/argbench/experiment/configs/prompting/prompting.json" \
--leaderboard-path "${DATA_PATH}/runs/prompting-$model-results.csv" --base_model "$model" --debug --train_subsample_amount 1

python -m  argbench.experiment.run -c "${CODE_PATH}/argbench/experiment/configs/prompting/prompting.json" \
--leaderboard-path "${DATA_PATH}/runs/prompting-$model-results.csv" --base_model "$model" --debug --train_subsample_amount 4
done

export TIME="$(sacct --format=Elapsed -j $SLURM_JOB_ID | tail -n 1 | xargs 2>&1)"
export JOB_ARGUMENTS="prompting;all-datasets;${model};\n"
echo "$SLURM_JOB_ID,$SLURM_JOB_NAME,$TIME,$JOB_ARGUMENTS" >> "$CODE_PATH/argbench/jobs/job-accounting.csv"
