#!/bin/bash -l
#SBATCH --job-name=open_debate_evidence
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output argbench/output/open_debate_evidence_output.out
#SBATCH --error argbench/output/open_debate_evidence_error.err
#SBATCH --gpus=1

export CODE_PATH="$BIGWORK/task-specific-argument-mining-and-generation"

module load Miniforge3
conda activate task-specific
python "$CODE_PATH/argbench/converter/convert_argument_summarization_open_debate_evidence_roush23.py"