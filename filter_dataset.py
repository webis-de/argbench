import os
from pathlib import Path

path_zenodo = "/bigwork/nhwpajjy/zenodo/tasks"
for dataset_path in os.listdir(path_zenodo):
    for file in os.listdir(Path(path_zenodo) / Path(dataset_path)):
        if "rate-0.1.json" not in file:
            path_file = Path(path_zenodo) / Path(dataset_path) / file
            os.remove(path_file)