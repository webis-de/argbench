#!/bin/bash
echo "${@:4}"
export CODE_PATH="$BIGWORK/task-specific-argument-mining-and-generation"
export CONFIG_PATH="$BIGWORK/task-specific-argument-mining-and-generation/argbench/experiment/configs/"
export DATA_PATH="$BIGWORK/task-specific-argument-mining-and-generation-data"
export gpu_count=$1
export experiment=$2
export jobname=$3

if [ "$gpu_count" == 1 ] ; then
sbatch <<EOT
#!/bin/bash -l
#SBATCH --job-name="$jobname"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=72:00:00
#SBATCH --output argbench/output/"$jobname"-%j.out
#SBATCH --error argbench/output/"$jobname"-%j.err
#SBATCH --gpus=1

module load Miniforge3
conda activate task-specific-new

python -m  argbench.experiment.run -c "${CONFIG_PATH}/in_task/${experiment}.json" ${@:4}
EOT
else
  echo "on accelerate"
sbatch <<EOT
#!/bin/bash -l
#SBATCH --job-name="$jobname"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=72:00:00
#SBATCH --output argbench/output/"$jobname"-%j.out
#SBATCH --error argbench/output/"$jobname"-%j.err
#SBATCH --gpus=${gpu_count}

module load Miniforge3
conda activate task-specific

accelerate launch --config_file "${CONFIG_PATH}/accelerate/config_${gpu_count}_gpus_2_stage.yaml" \\
-m  argbench.experiment.run -c "${CONFIG_PATH}/in_task/${experiment}.json" ${@:4}
EOT
fi;