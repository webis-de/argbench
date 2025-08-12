#!/bin/bash
export CODE_PATH="$BIGWORK/task-specific-argument-mining-and-generation"

sbatch <<EOT
#!/bin/bash -l
#SBATCH --job-name=test-job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=72:00:00
#SBATCH --output argbench/output/test-job-%j.out
#SBATCH --error argbench/output/test-job-%j.err


module load Miniforge3
conda activate task-specific
echo "working"
start=\$(date +%s)
python -c "[_ for i in range(10000000)]"
end=\$(date +%s)
export TIME=\$((end-start))
echo "$TIME,"jobname" >> "$CODE_PATH/argbench/jobs/job-accounting.csv"
EOT
