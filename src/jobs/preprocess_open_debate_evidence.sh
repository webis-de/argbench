#!/bin/bash -l
#SBATCH --job-name=task-specific
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=8
#SBATCH --mem=18G
#SBATCH --time=24:00:00
#SBATCH --output /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/src/output/open_debate_evidence_output.out
#SBATCH --error /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/src/output/open_debate_evidence_error.err
module load Miniforge3
conda activate task-specific
python /bigwork/nhwpajjy/task-specific-argument-mining-and-generation/src/converter/convert_argument_summarization_open_debate_evidence_roush23.py
