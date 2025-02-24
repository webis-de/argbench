#!/bin/bash -l
#SBATCH --job-name=task-specific
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --time=24:00:00
#SBATCH --output /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/output/hpo-stance-barhaim.out
#SBATCH --error /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/output/hpo-stance-barhaim.err
#SBATCH --gpus=a100:1
module load Miniforge3
conda activate task-specific
python /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/experiment/run.py -c /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/experiment/configs/hpo/barhaim17_hpo.json
