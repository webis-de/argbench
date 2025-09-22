cd "$CODE_PATH"
if [ -z "$dataset" ]; then
  datasets=$(jq '.test[]'  "${CODE_PATH}/argbench/experiment/configs/experiment_splits.json")
else
  datasets=("\"$dataset\"")
fi
for dataset in ${datasets[@]};
do
echo $dataset
done