#!/bin/bash

export model_list=argbench/data/models-small.txt

while getopts "abhvm:t:c:" opt; do
  case $opt in
    c)
      gpu_count=$OPTARG
    ;;
    b)
      model_list=argbench/data/models-large.txt
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

for model in "${models[@]}"; do
  for task in "${tasks[@]}"; do
    if [ -z "$validation" ]; then
      echo "test"
      if [ "$gpu_count" == 1 ] ; then
          bash argbench/jobs/in-tasks/run_skill_transfer_experiments.sh "$gpu_count" "$gpu_type" "${task}" "skill-transfer-$model-$task" --model "$model" --debug  --quantization --train_epochs 1 --skill_filter "$task"
        else
          bash argbench/jobs/in-tasks/run_skill_transfer_experiments.sh "$gpu_count" "$gpu_type" "${task}" "skill-transfer-$model-$task" --model "$model" --debug --optim "adamw_torch" --skill_filter "$task" --train_epochs 1 --skill_filter "$task"
        fi

    else
      echo "validation"
      if [ "$gpu_count" == 1 ] ; then
          bash argbench/jobs/in-tasks/run_skill_transfer_experiments.sh "$gpu_count" "$gpu_type" "${task}_hpo" "skill-transfer-$model-$task-hpo" --model "$model" --debug  --optim "adamw_torch" --sample --train_epochs 1 --skill_filter "$task"
        else
          echo "$gpu_count"
          echo "$gpu_type"
          bash argbench/jobs/in-tasks/run_skill_transfer_experiments.sh "$gpu_count" "$gpu_type" "${task}_hpo" "skill-transfer-$model-$task-hpo" --model "$model" --debug   --optim "adamw_torch" --sample --train_epochs 1 --skill_filter "$task"
        fi



     fi
    sleep 5
  done

done
