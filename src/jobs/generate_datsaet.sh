#!/bin/bash -l
#SBATCH --job-name=preprocessing-datasets
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/src/output/preprocessing-datasets.out
#SBATCH --error /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/src/output/preprocessing-datasets.err

module load Miniforge3
conda activate task-specific
for f in /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/src/converter/convert*.py;  do
python $f ;
echo $f
done
