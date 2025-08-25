from argbench.converter.common import tasks_path
from pathlib import Path
import pytest
import json
import os


curr_file = os.path.abspath(__file__)
config_file = Path(curr_file).parents[0] / "dataset_canonical_size.json"
with open(config_file) as f:
    DATASET_CANON_SIZE = json.load(f)


@pytest.mark.parametrize("dataset_size", DATASET_CANON_SIZE)
def test_dataset_canonical_size(dataset_size):
    with open(tasks_path() / "metadata.json", "r") as f:
        metadata = json.load(f)

    total_instances = 0
    for file in metadata[dataset_size["name"]]["file_list"]:
        if dataset_size.get("file_list"):
            if file not in dataset_size["file_list"]:
                continue
        file_path = tasks_path() / dataset_size["name"] / file
        with open(file_path, "r") as f:
            dataset = json.load(f)
            total_instances += len(dataset["Instances"])

    assert total_instances == dataset_size["canon_size"], f"Instance amount is not equal to canon instance amount: {dataset_size['name']}"
