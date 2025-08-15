#!/bin/bash
echo "${@:4}"
export CODE_PATH="$BIGWORK/task-specific-argument-mining-and-generation"
export CONFIG_PATH="$BIGWORK/task-specific-argument-mining-and-generation/argbench/experiment/configs/prompting/"
export DATA_PATH="$BIGWORK/task-specific-argument-mining-and-generation-data"
export gpu_count=$1
export experiment=$2
export jobname=$3
sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name="$jobname"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=72:00:00
#SBATCH --output argbench/output/"$jobname"-%j.out
#SBATCH --error argbench/output/"$jobname"-%j.err
#SBATCH --gpus="$gpu_count"

module load Miniforge3
conda activate new-env

start=\$(date +%s)

start_date=\$(date +%Y-%m-%d:%H:%m)
python -m  argbench.experiment.run -c "${CONFIG_PATH}/${experiment}.json" ${@:4}


end=\$(date +%s)

export Time=\$((end-start))
Time_HOURS=\$(awk -v t="\$Time" 'BEGIN { printf "%.2f", t / 3600 }')
Time_Minutes=\$(awk -v t="\$Time" 'BEGIN { printf "%.2f", t / 60 }')

echo "\$start_date,\$Time_HOURS,\$Time_Minutes,\$jobname" >> "$CODE_PATH/argbench/jobs/job-accounting.csv"
EOF
