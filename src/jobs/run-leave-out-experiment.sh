#!/bin/bash -l
#SBATCH --job-name=task-specific
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output output/output.out
#SBATCH --error errors/error.err
#SBATCH --gpus=1
module load Miniconda3
conda activate lang
#python /mnt/home/yajjour/task-specific-argument-mining-and-generation/src/experiment/process_jsonl.py -o /mnt/home/yajjour/task-specific-argument-mining-and-generation-data/
python /mnt/home/yajjour/task-specific-argument-mining-and-generation/src/experiment/run.py -c /mnt/home/yajjour/task-specific-argument-mining-and-generation/src/experiment/configs/complete_leave_one_out_ajjour17.json -co output.json
