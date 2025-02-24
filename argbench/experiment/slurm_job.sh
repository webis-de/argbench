#!/bin/bash --login
#SBATCH --job-name=web-login/sys/myjobs/default
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output test_serial-job_%j.out
#SBATCH --error test_serial-job_%j.err
#SBATCH --gres=gpu:a100m40:1

# run program
echo "I am running on $HOSTNAME"

nvidia-smi

pwd

cd /bigwork/nhwpbozd/task-specific-argument-mining-and-generation/src/experiment

module laod Miniconda3

conda activate exp

python run.py -c configs/alpaca_eval.json
