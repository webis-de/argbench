#!/bin/bash

export model_list=argbench/data/models-small.txt

while getopts "qabhvnsm:l:t:c:" opt; do
  case $opt in
    c)
      gpu_count=$OPTARG
    ;;
    b)
      model_list=argbench/data/models-large.txt
      ;;
    q)
      quantization="quantization"
      ;;
    l)
      model_adapter="$OPTARG"
    ;;
    n)
      no_training="no_training"
    ;;
      s)
        sample="sample"
      ;;
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
  mapfile -t models < "$model_list"
else
  models=("$model")
fi

if [ -n "$quantization" ]; then
    args+=("--quantization")
else
  args+=("--optim" "adamw_torch")
fi

if [ -n "$model_adapter" ]; then
    args+=("--model_id" "$model_adapter")
fi

if [ -n "$no_training" ]; then
  args+=("--train_epochs" 0)
fi

if [ -n "$sample" ]; then
  args+=("--sample")
fi


for model in "${models[@]}"; do
  for task in "${tasks[@]}"; do
    if [ -z "$validation" ]; then
      echo "test"
          bash argbench/jobs/in-tasks/run_in_tasks_experiments.sh "$gpu_count" "$gpu_type" "in_task_${task}" "in-tsk-$model-$task" --model "$model" --debug "${args[@]}" --max_length 1024
    else
      echo "validation"
          bash argbench/jobs/in-tasks/run_in_tasks_experiments.sh "$gpu_count" "$gpu_type" "in_task_${task}_hpo" "in-tsk-$model-$task-hpo" --model "$model"  --debug "${args[@]}" --max_length 1024
     fi
    sleep 5
  done

done
