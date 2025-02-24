#!/bin/bash -l
#SBATCH --job-name=preprocess
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --time=24:00:00
#SBATCH --output /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/output/preprocess-all-dataset.out
#SBATCH --error /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/output/preprocess-all-dataset.err
module load Miniforge3

conda activate task-specific

for f in /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/converter/convert*.py;  do
python $f ;
echo $f
done
python /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/experiment/preprocess.py -o /bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/tasks
