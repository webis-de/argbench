#!/bin/bash

export model_list=argbench/data/models-small.txt

while getopts "abhem:t:c:f:" opt; do
  case $opt in
    c)
      gpu_count="$OPTARG"
    ;;
    b)
      model_list=argbench/data/models-large.txt
      ;;
    a)
      gpu_type="a100"
    ;;
  e)
    error_analysis="error-analysis"
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
    f)
      file="$OPTARG"
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
  if [ -n "$file" ] ; then
    mapfile -t tasks < "$file"
  else
    mapfile -t tasks < argbench/data/target_tasks.txt
    fi
else
  tasks=("$task")
fi

if [ -z "$model" ]; then
  mapfile -t models < "$model_list"
else
  models=("$model")
fi

if [ -n "$error_analysis" ]; then
  args+=("--error_analysis")
fi
for model in "${models[@]}"; do
  for task in "${tasks[@]}"; do

    bash argbench/jobs/prompting/prompting.sh "$gpu_count" "$gpu_type" 24:00:00 prompting "pred-${task:0:6}-${model:0:6}-0-shot" --dataset "$task" --model "$model"  --sample "${args[@]}"
    sleep 5
    #bash argbench/jobs/prompting/prompting.sh "$gpu_count" "$gpu_type" 24:00:00 prompting "${task:0:6}-${model:0:6}-1-shot" --dataset "$task" --model "$model" -k 1  --sample "${args[@]}"
#    sleep 5
   bash argbench/jobs/prompting/prompting.sh "$gpu_count" "$gpu_type" 24:00:00 prompting "pred-${task:0:6}-${model:0:6}-4-shot" --dataset "$task" --model "$model" -k 4 --sample "${args[@]}"
    sleep 5
    bash argbench/jobs/prompting/prompting.sh "$gpu_count" "$gpu_type" 24:00:00 prompting "pred-${task}-${model}-cot" --dataset "$task" --model "$model" --cot --sample "${args[@]}"
    sleep 5
  done

done
