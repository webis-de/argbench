#!/bin/bash
echo "${@:5}"
export CODE_PATH="$BIGWORK/task-specific-argument-mining-and-generation"
export CONFIG_PATH="$BIGWORK/task-specific-argument-mining-and-generation/argbench/experiment/configs/"
export DATA_PATH="$BIGWORK/task-specific-argument-mining-and-generation-data"

export gpu_count=$1
export gpu_type=$2
export experiment=$3
export jobname=$4

if [ "$gpu_count" == 1 ] ; then
sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name="$jobname"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output argbench/output/"$jobname"-%j.out
#SBATCH --error argbench/output/"$jobname"-%j.err
#SBATCH --gres=gpu:"$gpu_type:$gpu_count"
#SBATCH --exclude=gpu004.kisski

module load GCCcore/.13.2.0
module load NVHPC/24.9-CUDA-12.6.0
module load Miniforge3
conda activate new-env
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
start=\$(date +%s)
Start_Date=\$(date +%Y-%m-%d:%H:%m)
echo "no parallel"
export CUDA_LAUNCH_BLOCKING=1
python -m  argbench.experiment.run -c "${CONFIG_PATH}/in_task/${experiment}.json" ${@:5} --max_len 1024
end=\$(date +%s)
export Time=\$((end-start))
TIME_HOURS=\$(awk -v t="\$Time" 'BEGIN { printf "%.2f", t / 3600 }')
Time_Minutes=\$(awk -v t="\$Time" 'BEGIN { printf "%.2f", t / 60 }')
Start_Date=\$(date +%Y-%m-%d:%H:%m)
echo "\$Start_Date,\$Time_HOURS,\$Time_Minutes,$jobname" >> "$CODE_PATH/argbench/jobs/job-accounting.csv"

EOF
else
  echo "on accelerate"
sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name="$jobname"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=72:00:00
#SBATCH --output argbench/output/"$jobname"-%j.out
#SBATCH --error argbench/output/"$jobname"-%j.err
#SBATCH --gres=gpu:"$gpu_type:$gpu_count"
#SBATCH --exclude=gpu004.kisski

module load GCCcore/.13.2.0
module load NVHPC/24.9-CUDA-12.6.0

module load Miniforge3
conda activate new-env
start=\$(date +%s)
Start_Date=\$(date +%Y-%m-%d:%H:%m)
echo "parallel"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_LAUNCH_BLOCKING=1
accelerate launch --config_file "${CONFIG_PATH}/accelerate/config_${gpu_count}_gpus_3_stage.yaml" \\
-m  argbench.experiment.run -c "${CONFIG_PATH}/in_task/${experiment}.json" ${@:5}
end=\$(date +%s)
export Time=\$((end-start))
TIME_HOURS=\$(awk -v t="\$Time" 'BEGIN { printf "%.2f", t / 3600 }')
Time_Minutes=\$(awk -v t="\$Time" 'BEGIN { printf "%.2f", t / 60 }')

echo "\$Start_Date,\$TIME_HOURS,\$Time_Minutes,$jobname" >> "$CODE_PATH/argbench/jobs/job-accounting.csv"

EOF
fi;