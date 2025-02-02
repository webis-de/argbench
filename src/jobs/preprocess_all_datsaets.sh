#!/bin/bash -l
#SBATCH --job-name=preprocess
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/src/output/preprocess-all-dataset.out
#SBATCH --error /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/src/output/preprocess-all-dataset.err
module load Miniconda3
conda activate task-specific
python /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/src/experiment/preprocess.py -o /bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/