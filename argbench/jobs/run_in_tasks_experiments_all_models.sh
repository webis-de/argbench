#!/bin/bash -l
models=("gemma-9-2b-it" "gemma-2-27b-it")
#models=("gemma-9-2b-it" "gemma-2-27b-it" "mistral-7b-inst-3" "mistral-small" "deepseek-r1-distill-1.5b" "deepseek-r1-distill-32b")

for model in ${models[@]};
do
sbatch argbench/jobs/run_in_tasks_experiments.sh "$model" --output "argbench/output/in-task-hpo-experiment-$model.out" --error "argbench/output/in-task-hpo-experiment-$model.errr"
done