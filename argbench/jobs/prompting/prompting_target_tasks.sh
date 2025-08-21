#!/bin/bash


while getopts "m:t:" opt; do
  case $opt in
    m)
      model="$OPTARG"
      ;;
    t)
      task="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      # This case handles an argument that is present but has no value
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

shift $((OPTIND-1))

if [ -z "$task" ]; then
  mapfile -t tasks < argbench/jobs/task_sample.txt
else
  tasks=("$task")
fi

if [ -z "$model" ]; then
  models=("phi-3.5-moe-7.5b" "qwen3-4b" "deepseek-r1-distill-7b" "llama-3.1-8b-instruct" "gemma3-4b")
else
  models=("$model")
fi

for model in "${models[@]}"; do
  for task in "${tasks[@]}"; do

    bash argbench/jobs/prompting/prompting.sh 1 prompting "${task:0:3}-${model:0:3}-0-shot" --seed 17023 --dataset "$task" --model "$model"
    bash argbench/jobs/prompting/prompting.sh 1 prompting "${task:0:3}-${model:0:3}-1-shot" --seed 17023 --dataset "$task" --model "$model" -k 1
    bash argbench/jobs/prompting/prompting.sh 1 prompting "${task:0:3}-${model:0:3}-4-shot" --seed 17023 --dataset "$task" --model "$model" -k 4
    bash argbench/jobs/prompting/prompting.sh 1 prompting "${task:0:3}-${model:0:3}-cot" --seed 17023 --dataset "$task" --model "$model" --cot
  done

done
