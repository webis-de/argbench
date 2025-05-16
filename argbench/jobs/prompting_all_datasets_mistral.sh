#!/bin/bash -l
skill=$1

datasets=$(jq  --arg s "$skill" '.[$s][]' argbench/jobs/skills.json)

for dataset in ${datasets[@]};
do
echo $dataset
sbatch argbench/jobs/prompting_dataset.sh "$dataset" "mistral-7b-inst-3"
sleep 30
done