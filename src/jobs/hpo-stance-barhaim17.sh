#!/bin/bash -l
#SBATCH --job-name=task-specific
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/src/output/hpo-stance-barhaim.out
#SBATCH --error /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/src/output/hpo-stance-barhaim.err
#SBATCH --gpus=a100:1
module load Miniconda3
conda activate task-specific

python /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/src/experiment/run.py -c /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/src/experiment/configs/hpo/complete_leave_one_out_barhaim17_mistral_hpo_test.json
