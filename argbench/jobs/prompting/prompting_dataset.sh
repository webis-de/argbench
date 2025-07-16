#!/bin/bash
sbatch <<EOT
#!/bin/bash -l
#SBATCH --job-name=prmt-dataset
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=72:00:00
#SBATCH --output argbench/output/prmt-dataset-%j.out
#SBATCH --error argbench/output/prmt-dataset-%j.err
#SBATCH --gpus=$3


module load Miniforge3
conda activate task-specific

model=$2
dataset=$1

echo "working on $dataset"

export CODE_PATH="$BIGWORK/task-specific-argument-mining-and-generation"
export DATA_PATH="$BIGWORK/task-specific-argument-mining-and-generation-data"

if [ -z "$model" ]; then


models=("gemma-2-9b-it" "gemma-2-27b-it" "mistral-7b-inst-3" "mistral-small" "deepseek-r1-distill-7b" "deepseek-r1-distill-32b")
else

models=($model)
fi

cd "$CODE_PATH"

for model in ${models[@]};
do
echo "using the model $model"
python -m  argbench.experiment.run -c "${CODE_PATH}/argbench/experiment/configs/prompting/prompting.json" \
--leaderboard-path "${DATA_PATH}/runs/prompting-$model-complete-results.csv" --base_model "$model" --test_dataset_name "$dataset"
done

export TIME="$(sacct --format=Elapsed -j $SLURM_JOB_ID | tail -n 1 | xargs 2>&1)"
export JOB_ARGUMENTS="prompting;${dataset};${model};\n"
echo "$SLURM_JOB_ID,$SLURM_JOB_NAME,$TIME,$JOB_ARGUMENTS" >> "$CODE_PATH/argbench/jobs/job-accounting.csv"
EOT