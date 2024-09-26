from argparse import ArgumentParser
import json
from pathlib import Path
from torch import _convert_indices_from_coo_to_csr
from yaml import load, Loader
import ndjson
import os
import pandas as pd

class PandasDataset:

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return self.dataframe.iloc[idx].to_dict()

def data_repo():
    """Get path of data repository"""
    curr_file = os.path.abspath(__file__)
    config_file = Path(curr_file).parents[2] / "config.yaml"
    with open(config_file, "r") as f:
        config = load(f, Loader=Loader)
        return Path(config["data_repo"])

def tasks_path():
    """Path to tasks folder"""
    dataset_folder = data_repo() / "tasks"
    return dataset_folder

def get_metadata():
    with open(tasks_path() / "metadata.json", "r") as f:
        return json.load(f)

def collect_files(
        task_path,
        metadata,
        test_configs,
        include_genre=None,
        include_subarea=None,
        include_task=None,
        is_leave_one_out=False,
        is_evaluate=False,
        exclude_datasets=None
):
    if include_genre:
        include_genre = set(include_genre)
    if include_subarea:
        include_subarea = set(include_subarea)
    if not include_task:
        include_task = []
    if not exclude_datasets:
        exclude_datasets = []


    test_files = []
    for file in metadata[test_configs["name"]]["file_list"]:
        if test_configs["match"] in file:
            test_files.append(task_path / test_configs["name"] / file)

    if is_evaluate:
        return [], test_files

    train_files = []
    for task in os.listdir(task_path):
        task_data_path = task_path / task

        if task in exclude_datasets:
            continue
        if not os.path.isdir(task_data_path):
            continue

        if task not in metadata:
            print(f"{task} not in metadata!")
            continue
        genres = metadata[task].get("genre", set())
        subareas = metadata[task].get("subareas", set())

        for task_file in os.listdir(task_data_path):
            task_file_path = task_data_path / task_file
            if not os.path.isfile(task_file_path):
                continue
            if task_file_path in test_files:
                continue

            task_select = False
            if include_task:
                select_task = next((task_file for t_s in include_task if t_s in task_file), None)
                if select_task:
                    train_files.append(task_file_path)

            if is_leave_one_out:
                train_files.append(task_file_path)
                continue

            if include_genre and include_genre.intersection(genres):
                train_files.append(task_file_path)
                continue

            if include_subarea and include_subarea.intersection(subareas):
                train_files.append(task_file_path)

    return train_files, test_files


def compile_datasets(dataset_paths, prompt_template, subsample_amount=None, subsample_rate=None):
    """
    Compile all datasets into a dataframe
    """
    def template_formatter(row):
        return prompt_template.format(
            instance_input=row["input"],
            definition=row["definition"])

    datasets = []
    for task_path in dataset_paths:
        print(task_path)
        task_data = pd.read_json(task_path, lines=True)

        if subsample_amount:
            task_data = task_data.sample(subsample_amount, axis=0)
        if subsample_rate:
            task_data = task_data.sample(frac=subsample_rate, axis=0)

        task_data["input"] = task_data.apply(template_formatter, axis=1)

        datasets.append(task_data[["id", "input", "output"]])

    if datasets:
        return pd.concat(datasets, axis=0).reset_index(drop=True)
    return pd.DataFrame()


def collect_datasets(run_config):
    """
    Collect all datasets to use in run
    """
    train_config = run_config.train_datasets
    test_config = run_config.test_datasets
    tasks_path = Path(run_config.data_folder)

    metadata = get_metadata()

    train_tasks, test_tasks = collect_files(
        tasks_path,
        metadata,
        test_config,
        train_config.get("include_genre"),
        train_config.get("include_subarea"),
        train_config.get("include_task"),
        train_config.get("leave_one_out", False),
        run_config.is_eval,
        train_config.get("exclude_datasets")
    )

    print("Train datasets collected:")
    train_dataset = compile_datasets(
        train_tasks,
        train_config["prompt_template"],
        train_config.get("subsample_amount", None),
        train_config.get("subsample_rate", None)
    )
    print("Test datasets collected:")
    test_dataset = compile_datasets(
        test_tasks,
        test_config["prompt_template"],
        test_config.get("subsample_amount", None),
        test_config.get("subsample_rate", None)
    )

    return train_dataset, test_dataset
