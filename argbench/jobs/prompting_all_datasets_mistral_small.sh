#!/bin/bash -l
skill=$1

datasets=$(jq  --arg s "$skill" '.[$s][]' argbench/jobs/skills.json)

for dataset in ${datasets[@]};
do

sbatch argbench/jobs/prompting_dataset.sh "$dataset" "mistral-small"
sleep 30
done