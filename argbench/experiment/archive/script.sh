datasets=$(jq '.validation[]'  "argbench/experiment/configs/experiment_splits.json")
for dataset in ${datasets[@]};
do
echo $dataset
python -m  argbench.experiment.run -c "${CODE_PATH}/argbench/experiment/configs/hpo/in_task_hpo.json" \
--model "$model" --dataset "$dataset"

done
