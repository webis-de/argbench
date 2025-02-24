#!/bin/bash -l
#SBATCH --job-name=task-specific
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/output/output.out
#SBATCH --error /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/output/error.err
#SBATCH --gpus=1
module load Miniconda3
conda activate task-specific
python /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/experiment/preprocess.py -o /mnt/home/yajjour/task-specific-argument-mining-and-generation-data/
python /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/experiment/run.py -c /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/experiment/configs/config_leave_one_out.json -c /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/experiment/configs/config_leave_one_out_ajjour17.json
