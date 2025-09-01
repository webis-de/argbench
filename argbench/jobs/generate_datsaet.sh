# set the right bigwork to either kisski or luis
export BIGWORK=/bigwork/nhwpajjy/
#export BIGWORK=/mnt/home/yajjour
echo "here is bigwork $BIGWORK"

export DATA_PATH="$BIGWORK/task-specific-argument-mining-and-generation-data"
echo "here is data path $DATA_PATH"
python "argbench/converter/convert_$1.py"
python -m argbench.experiment.preprocess -t "$1" -o "${DATA_PATH}/tasks"
