#!/bin/bash


while getopts "ahvm:t:" opt; do
  case $opt in
    a)
      gpu_type="a100"
    ;;
    h)
      gpu_type="h100"
    ;;
    v)
     validation="validation"
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
  tasks=("generation" "mining" "quality" "perspective" "reasoning")
else
  tasks=("$task")
fi

if [ -z "$model" ]; then
  models=("phi-3.5-moe-7.6b" "qwen3-4b" "deepseek-r1-distill-7b" "llama-3.1-8b-instruct" "gemma-3-4b-it" "mistral-7b-inst-3" "qwen3-1.7b" "deepseek-r1-distill-1.5b")
else
  models=("$model")
fi

for model in "${models[@]}"; do
  for task in "${tasks[@]}"; do
    if [ -z "$validation" ]; then
      echo "test"
      bash argbench/jobs/in-tasks/run_in_tasks_experiments.sh 1 "$gpu_type" "in_task_${task}" "in-tsk-$model-$task" --model "$model" --debug  --quantization
    else
      echo "validation"
      bash argbench/jobs/in-tasks/run_in_tasks_experiments.sh 1 "$gpu_type" "in_task_${task}_hpo" "in-tsk-$model-$task-hpo" --model "$model" --debug  --quantization --sample

     fi
    sleep 5
  done

done
