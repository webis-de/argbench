#!/bin/bash -l
#SBATCH --job-name=hpo-stance
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=12
#SBATCH --mem=15G
#SBATCH --time=96:00:00
#SBATCH --output argbench/output/hpo-stance-barhaim.out
#SBATCH --error argbench/output/hpo-stance-barhaim.err
#SBATCH --gpus=a100:1

export CODE_PATH="$BIGWORK/task-specific-argument-mining-and-generation"
module load Miniforge3
conda activate task-specific

python -m  argbench.experiment.run  -c "$CODE_PATH/argbench/experiment/configs/hpo/barhaim17_mistral_hpo.json"


export TIME="$(sacct --format=Elapsed -j $SLURM_JOB_ID | tail -n 1 | xargs 2>&1)"
export JOB_ARGUMENTS="fine-tuning-hpo;barhaim-stance;mistral;\n"
echo "$SLURM_JOB_ID,$SLURM_JOB_NAME,$TIME,$JOB_ARGUMENTS" >> "$CODE_PATH/argbench/jobs/job-accounting.csv"