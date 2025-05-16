#!/bin/bash -l
skill=$1
model=$2
datasets=$(jq  --arg s "$skill" '.[$s][]' argbench/jobs/skills.json)



for dataset in ${datasets[@]};
do
  dataset=${dataset:1:-1}
echo "$model,$dataset,zero-shot" > "argbench/jobs/jobs-started.txt"
sbatch argbench/jobs/prompting_dataset.sh "$dataset" "$model" >> "argbench/jobs/jobs-started.txt"
sleep 30

echo "$model,$dataset,cot" > "argbench/jobs/jobs-started.txt"
sbatch argbench/jobs/prompting_dataset_cot.sh "$dataset" "$model" >> "argbench/jobs/jobs-started.txt"
sleep 30

echo "$model,$dataset,one-shot" > "argbench/jobs/jobs-started.txt"
sbatch argbench/jobs/prompting_dataset_one_shot.sh "$dataset" "$model" >> "argbench/jobs/jobs-started.txt"
sleep 30

echo "$model,$dataset,four-shot" > "argbench/jobs/jobs-started.txt"
sbatch argbench/jobs/prompting_dataset_four_shot.sh "$dataset" "$model" >> "argbench/jobs/jobs-started.txt"
sleep 30

done