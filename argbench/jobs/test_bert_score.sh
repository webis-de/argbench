#!/bin/bash -l
#SBATCH --job-name=open_debate_evidence
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=8
#SBATCH --mem=18G
#SBATCH --time=24:00:00
#SBATCH --output argbench/output/test_bert_score.out
#SBATCH --error argbench/output/test_bert_score.err
#SBATCH --gres=gpu:a100:1

export CODE_PATH="$BIGWORK/task-specific-argument-mining-and-generation"

module load Miniforge3
conda activate new-env
python -m unittest argbench.experiment.tests.test_bert_score
