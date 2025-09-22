import os.path

import pandas as pd
from datasets import load_from_disk
from argbench.experiment.config import *
datasets =load_from_disk("/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/argbench-dataset/argbench-prompting-zero-shot-small")
path_metadata = "/bigwork/nhwpajjy/computational-argumentation-tasks-instructions/tasks/metadata.json"

with open(path_metadata) as file:
    metadata = json.load(file)

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), Path("../data/prompting-set-stats.csv"))
skills_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), Path("../data/skills-stats.csv"))
dataset_sizes = []


for dataset in datasets:
    dataset_name = dataset.replace("test_", "")
    dataset_size = {}
    dataset_size["task"] = dataset_name
    dataset_size["test-sampled-size"] = datasets[dataset].shape[0]
    dataset_size["skill"] = metadata[dataset_name]["skill"]
    dataset_sizes.append(dataset_size)
print(dataset_sizes)
df = pd.DataFrame.from_records(dataset_sizes)
df.to_csv(data_path, sep=",", index=False)
df.groupby("skill").agg({"test-sampled-size":"sum"}).reset_index().to_csv(skills_path, sep=",", index=False)


