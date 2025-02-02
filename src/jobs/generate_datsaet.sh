#!/bin/bash -l

python "../converter/convert_$1.py"
python "../experiment/preprocess.py" -t "$1" -o /bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/tasks
