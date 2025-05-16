#!/bin/bash -l
models=("gemma-2-9b-it"  "mistral-7b-inst-3" "deepseek-r1-distill-7b" "gemma-2-27b-it" "mistral-small")

dataset=$1

for model in ${models[@]};
do
sbatch argbench/jobs/run_in_tasks_experiments.sh "$model" "$dataset"
sleep 5
done