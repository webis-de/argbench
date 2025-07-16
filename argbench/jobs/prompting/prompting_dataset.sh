#!/bin/bash
echo "$1 $2 $3"
export CODE_PATH="$BIGWORK/task-specific-argument-mining-and-generation"
export DATA_PATH="$BIGWORK/task-specific-argument-mining-and-generation-data"
export model=$2
export dataset=$1

sbatch <<EOT
#!/bin/bash -l
#SBATCH --job-name=prmt-dataset
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=72:00:00
#SBATCH --output argbench/output/prmt-dataset-%j.out
#SBATCH --error argbench/output/prmt-dataset-%j.err
#SBATCH --gpus="$3"


module load Miniforge3
conda activate task-specific


python -m  argbench.experiment.run -c "${CODE_PATH}/argbench/experiment/configs/prompting/prompting.json" \
--leaderboard-path "${DATA_PATH}/runs/prompting-$model-complete-results.csv" --base_model "$model" --test_dataset_name "$dataset" --debug

EOT
