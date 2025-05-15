#!/bin/bash -l
datasets=$(cat tasks)

for dataset in ${datasets[@]};
do

sbatch argbench/jobs/prompting_dataset.sh "$dataset" "gemma-2-9b-it"
done