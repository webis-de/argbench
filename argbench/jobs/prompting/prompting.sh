#!/bin/bash
echo "${@:2}"
export CODE_PATH="$BIGWORK/task-specific-argument-mining-and-generation"
export DATA_PATH="$BIGWORK/task-specific-argument-mining-and-generation-data"
export gpu_count=$1
sbatch <<EOT
#!/bin/bash -l
#SBATCH --job-name=prmt
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=72:00:00
#SBATCH --output argbench/output/prmt-%j.out
#SBATCH --error argbench/output/prmt-%j.err
#SBATCH --gpus="$gpu_count"


module load Miniforge3
conda activate task-specific


python -m  argbench.experiment.run -c "${CODE_PATH}/argbench/experiment/configs/prompting/prompting.json" \
 ${@:2}

EOT
