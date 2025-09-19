#!/bin/bash

export model_list=argbench/data/models-small.txt

while getopts "abhm:c:" opt; do
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
    h)
      gpu_type="h100"
    ;;

    m)
      model="$OPTARG"
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



if [ -z "$model" ]; then
  mapfile -t models < "$model_list"
else
  models=("$model")
fi

for model in "${models[@]}"; do
    bash argbench/jobs/prompting/prompting.sh "$gpu_count" "$gpu_type" prompting "${model:0:6}-0-shot"  --model "$model" --sample --seed 1517
    sleep 5
    bash argbench/jobs/prompting/prompting.sh "$gpu_count" "$gpu_type" prompting "${model:0:6}-4-shot"  --model "$model" -k 4 --sample --seed 1517
    sleep 5
    bash argbench/jobs/prompting/prompting.sh "$gpu_count" "$gpu_type" prompting "${model:0:6}-cot" --model "$model" --cot --sample --seed 1517
    sleep 5
done
