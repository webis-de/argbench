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

export CONFIG_PATH="$BIGWORK/task-specific-argument-mining-and-generation/argbench/experiment/configs/prompting/prompting.json"

#module load Miniforge3
#conda activate new-env
#
#for f in "${CODE_PATH}"/argbench/converter/convert*.py;  do
#python $f ;
#echo $f
#done
#cd "$CODE_PATH"
#python -m argbench.experiment.preprocess -o "$DATA_PATH/tasks"
#find "$DATA_PATH/tasks/" -name "*size*"  | xargs -I % rm %
#find "$DATA_PATH/tasks/" -name "*rate*"  | xargs -I % rm %

python -m  argbench.experiment.prepare_experiment -c "$CODE_PATH"