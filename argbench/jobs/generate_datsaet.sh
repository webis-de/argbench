# Does not work for s
if [ -e "/bigwork/nhwpajjy/" ]; then
  export BIGWORK=/bigwork/nhwpajjy/
  else
  export BIGWORK=/mnt/home/yajjour
fi
echo "here is bigwork $BIGWORK"

export DATA_PATH="$BIGWORK/task-specific-argument-mining-and-generation-data"
echo "here is data path $DATA_PATH"
python "argbench/converter/convert_$1.py"
python -m argbench.experiment.preprocess -t "$1" -o "${DATA_PATH}/tasks"