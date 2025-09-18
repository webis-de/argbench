#!/bin/bash


while getopts "ahm:" opt; do
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
  models=("phi-3.5-moe-7.6b" "qwen3-4b" "deepseek-r1-distill-7b" "llama-3.1-8b-instruct")
else
  models=("$model")
fi

for model in "${models[@]}"; do
#    bash argbench/jobs/prompting/prompting.sh 1 "$gpu_type" prompting "${model:0:3}-0-shot"  --model "$model" --debug --sample --seed 1517
#    sleep 5
#    bash argbench/jobs/prompting/prompting.sh 1 "$gpu_type" prompting "${model:0:3}-4-shot"  --model "$model" -k 4 --debug --sample --seed 1517
#    sleep 5
    bash argbench/jobs/prompting/prompting.sh 1 "$gpu_type" prompting "${model:0:3}-cot" --model "$model" --cot --debug --sample --seed 1517
    sleep 5
done
