#!/bin/bash

export CODE_PATH="$BIGWORK/task-specific-argument-mining-and-generation"
sbatch << EOF
#!/bin/bash -l
#SBATCH --job-name=test-job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=72:00:00
#SBATCH --output argbench/output/test-job-%j.out
#SBATCH --error argbench/output/test-job-%j.err
#SBATCH --partition tnt

module load Miniforge3
conda activate task-specific
start=\$(date +%s)
python -c "[print(i) for i in range(1000000000)]"
end=\$(date +%s)
start_date=date +%Y-%m-%d:%H:%m
export Time=\$((end-start))

Time=\$(awk -v t="\$Time" 'BEGIN { printf "%.2f", t / 3600 }')
echo "\$start_date,\$Time,jobname" >> "$CODE_PATH/argbench/jobs/job-accounting.csv"


EOF
