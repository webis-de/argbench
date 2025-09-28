#!/bin/bash
#/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/training_output/qwen3-4b-hpo-cross-task-stance_classification_ukp_sentential_stab18-09-17-15:19:54

export model_list=argbench/data/models-small.txt
while getopts "qabhnvm:l:t:c:g:" opt; do
  case $opt in
    c)
      gpu_count="$OPTARG"
    ;;
    b)
      model_list=argbench/data/models-large.txt
      ;;
    q)
      quantization="quantization"
      ;;

    a)
      gpu_type="a100"
    ;;
    l)
      model_adapter="$OPTARG"
    ;;
    n)
      no_training="no_training"
    ;;
    g)
     goal_task="$OPTARG"
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

args=()

# If QUANTIZATION is set, add the flag to the array
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

# Run the command with the array expanded
# The command will be executed as: ./your_command
if [ -n "$goal_task" ]; then
  case $goal_task in
  mining)
    goal_task="argument_unit_segmentation_webDiscourse_ajjour17"
    ;;
  generation)
    goal_task="counter_argument_generation_cmv_hua18"
  ;;
  reasoning)
    goal_task="fallacy_detection_cmv_adhominem_habernal18"
    ;;
  quality)
    goal_task="argument_rating_dagstuhl_15512_wachsmuth17"
    ;;
  perspective)
    goal_task="argument_similarity_ukp_aspect_reimers19"
    ;;
  esac
  args+=("--dataset" "$goal_task")
fi

for model in "${models[@]}"; do
  for task in "${tasks[@]}"; do
    if [ -z "$validation" ]; then
      echo "test"
        bash argbench/jobs/cross-tasks/run_cross_tasks_experiment.sh "$gpu_count" "$gpu_type" "cross_task" "cross-tsk-$model-$task" --model "$model"  --debug --sample "${args[@]}"
    else
        bash argbench/jobs/cross-tasks/run_cross_tasks_experiment.sh "$gpu_count" "$gpu_type" "cross_task_val_hpo" "cross-tsk-$model-hpo" --model "$model"  --debug "${args[@]}"
     fi
    sleep 5
  done

done


