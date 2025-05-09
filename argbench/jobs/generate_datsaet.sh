# Does not work for s
if [ -e "$file" ]; then
  BIGWORK=/bigwork/nhwpajjy/
  else
  BIGWORK=/mnt/hme/yajjour
fi


export DATA_PATH="$BIGWORK/task-specific-argument-mining-and-generation-data"
python "argbench/converter/convert_$1.py"
python -m argbench.experiment.preprocess -t "$1" -o "${DATA_PATH}/tasks"