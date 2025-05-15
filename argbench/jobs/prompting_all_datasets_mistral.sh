#!/bin/bash -l
datasets=$(cat tasks)

for dataset in ${datasets[@]};
do

sbatch argbench/jobs/prompting_dataset.sh "$dataset" "mistral-7b-inst-3"
done