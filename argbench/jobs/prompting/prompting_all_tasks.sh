#!/bin/bash


while getopts "m:" opt; do
  case $opt in
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
  models=("phi-3.5-moe-7.5b" "qwen3-4b" "deepseek-r1-distill-7b" "llama-3.1-8b-instruct" "gemma3-4b")
else
  models=("$model")
fi

for model in "${models[@]}"; do
    bash argbench/jobs/prompting/prompting.sh 1 prompting "${model:0:3}-0-shot" --max-length 128000 --model "$model" --debug --sample
    sleep 5
    bash argbench/jobs/prompting/prompting.sh 1 prompting "${model:0:3}-1-shot" --max-length 128000  --model "$model" -k 1 --debug --sample
    sleep 5
    bash argbench/jobs/prompting/prompting.sh 1 prompting "${model:0:3}-4-shot" --max-length 128000  --model "$model" -k 4 --debug --sample
    sleep 5
    bash argbench/jobs/prompting/prompting.sh 1 prompting "${model:0:3}-cot" --max-length 128000 --model "$model" --cot --debug --sample
    sleep 5
done
