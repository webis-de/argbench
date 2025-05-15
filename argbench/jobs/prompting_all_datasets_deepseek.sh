#!/bin/bash -l
datasets=$(cat tasks)

for dataset in ${datasets[@]};
do

sbatch argbench/jobs/prompting_dataset.sh "$dataset" "deepseek-r1-distill-7b"
done