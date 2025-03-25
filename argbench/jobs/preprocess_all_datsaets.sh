#!/bin/bash -l
#SBATCH --job-name=preprocess-all-dataset
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=24:00:00
#SBATCH --output argbench/output/preprocess-all-dataset.out
#SBATCH --error argbench/output/preprocess-all-dataset.err


export CODE_PATH="$BIGWORK/task-specific-argument-mining-and-generation"
export DATA_PATH="$BIGWORK/task-specific-argument-mining-and-generation-data"


module load Miniforge3
conda activate task-specific

for f in "${CODE_PATH}/argbench/converter/convert*.py";  do
python $f ;
echo $f
done
cd "$CODE_PATH"
python -m argbench.experiment.preprocess -o "$DATA_PATH/tasks"
