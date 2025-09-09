#!/bin/bash


while getopts "ahm:t:" opt; do
  case $opt in
    a)
      gpu_type="a100"
    ;;
    h)
      gpu_type="h100"
    ;;

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


if [ -z "$task" ]; then
  mapfile -t tasks < argbench/data/target_tasks.txt
else
  tasks=("$task")
fi

if [ -z "$model" ]; then
  models=("phi-3.5-moe-7.5b" "qwen3-4b" "deepseek-r1-distill-7b" "llama-3.1-8b-instruct" "gemma3-4b" "mistral-7b-inst-3")
else
  models=("$model")
fi

for model in "${models[@]}"; do
  for task in "${tasks[@]}"; do

    bash argbench/jobs/prompting/prompting.sh 1 "$gpu_type" prompting "${task:0:3}-${model:0:3}-0-shot" --dataset "$task" --model "$model" --debug --sample
    sleep 5
#    bash argbench/jobs/prompting/prompting.sh 1 "$gpu_type" prompting "${task:0:3}-${model:0:3}-1-shot" --dataset "$task" --model "$model" -k 1 --debug --sample
#    sleep 5
#    bash argbench/jobs/prompting/prompting.sh 1 "$gpu_type" prompting "${task:0:3}-${model:0:3}-4-shot" --dataset "$task" --model "$model" -k 4 --debug --sample
#    sleep 5
#    bash argbench/jobs/prompting/prompting.sh 1 "$gpu_type" prompting "${task:0:3}-${model:0:3}-cot" --dataset "$task" --model "$model" --cot --debug --sample
#    sleep 5
  done

done
