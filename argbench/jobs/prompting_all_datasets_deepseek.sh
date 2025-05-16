#!/bin/bash -l
skill=$1

datasets=$(jq  --arg s "$skill" '.[$s][]' argbench/jobs/skills.json)

for dataset in ${datasets[@]};
do

sbatch argbench/jobs/prompting_dataset.sh "$dataset" "deepseek-r1-distill-7b"
sleep 30
done