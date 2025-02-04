import json
import logging
import datasets
import ndjson
import os
import pandas as pd


from collections import defaultdict
from pathlib import Path
from IPython.core.debugger import set_trace
from datasets import DatasetDict, Dataset

from yaml import load, Loader
from utils import get_logger


logger = get_logger(__name__)




class PandasDataset:
    """
    Class to convert pandas DataFrame into usable Dataset
    """
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
    """Get metadata file"""
    with open(tasks_path() / "metadata.json", "r") as f:
        return json.load(f)

def collect_files(
        task_path,
        metadata,
        test_configs,
        is_prompting=False,
):
    """
    Collects files for train and test sets

    :param task_path: path to folder with task data
    :param metadata: metadata dictionary
    :param test_configs: test dataset configuration dict

    :param is_prompting: Should only test set be retruned for evaluation

    :returns: Tuple of train dataset files and test dataset files
    """
    #set_trace()

    test_task = test_configs["tasks"]

    test_files = {}

    test_files[test_task] = []
    for file in metadata[test_task]["file_list"]:
        test_files[test_task].append(
            task_path / test_task / file)


    if is_prompting:
        return {}, test_files

    train_files = defaultdict(list)
    for task in os.listdir(task_path):
        task_data_path = task_path / task

        if not os.path.isdir(task_data_path):
            logger.warning(f"{task} not found!")
            continue

        if task not in metadata:
            logger.warning(f"{task} not in metadata!")
            continue

        logger.debug(f"reading {task_data_path}")

        for task_file in os.listdir(task_data_path):
            task_file_path = task_data_path / task_file
            logger.info(f"adding {task_file_path} to training")

            if not os.path.isfile(task_file_path) or task_file_path in test_files:
                continue

            train_files[task].append(task_file_path)
    return train_files, test_files


def compile_datasets(
        task_datasets,
        prompt_template,
        subsample_amount=None,
        subsample_rate=None,
        filetype="ndjson", training=True):
    """
    Read dataset file and compile all datasets into one dataframe

    :param task_datasets: List of dataset file paths for each dataset
    :param prompt_template: Template to compile dataset variables into prompt
    :param subsample_amount: Amount of samples to take from dataset file
    :param subsample_rate: % of instances to take from dataset
    :returns: Full compiled DataFrame of all datasets instances
    """
    def template_formatter(row):
        return prompt_template.format(
            instance_input=row["document"],
            definition=row["definition"],
            example=row["example"]
        )

    test_datasets = []
    training_datasets = []
    for dataset in task_datasets:
        total_datasets = []
        for task_path in task_datasets[dataset]:
            logger.info(f"reading {task_path}")
            if filetype == "ndjson":
                total_datasets.append(pd.read_json(task_path, lines=True))
            elif filetype == "parquet":
                total_datasets.append(pd.read_parquet(task_path))

        task_data = pd.concat(total_datasets, axis=0).reset_index(drop=True)

        if subsample_amount:
            task_data = task_data.sample(subsample_amount, axis=0)
        if subsample_rate:
            task_data = task_data.sample(frac=subsample_rate, axis=0)

        example_record = task_data.sample(n=1)

        task_data = task_data[~task_data.index.isin(example_record.index)]
        #set_trace()
        example_instance = f'Input: {example_record["input"].values[0]}\nOutput: {example_record["output"].values[0]}'

        task_data.rename(columns={"input": "document"}, inplace=True)
        task_data["example"] = example_instance

        task_data["input"] = task_data.apply(template_formatter, axis=1)
        for column in task_data.columns:
            task_data[column] = task_data[column].astype(str)

        if training:
            task_data["task"] = dataset
            task_df = task_data[["id","document", "input", "output", "task"]]
        else:
            task_df = task_data[["id","document", "input", "output", "task"]]


        if training:
            training_datasets.append(task_df)
        else:
            test_datasets.append(task_df)
#

    if training:
        if len(training_datasets):
            all_training_df = pd.concat(training_datasets, axis=0).reset_index(drop=True)
        else:
            all_training_df = pd.DataFrame()
        logger.info(all_training_df.info())
        return Dataset.from_pandas(all_training_df)
    else:
        all_test_df = pd.concat(test_datasets)
        return Dataset.from_pandas(all_test_df)


def collect_datasets(run_config):
    """
    Use RunConfig to create train and test datasets

    :param run_config: RunConfig with train_datasets and test_datasets config dicts
    :returns: Tuple of train and test datasets in pandas DataFrame
    """
    train_config = run_config.train_datasets
    test_config = run_config.test_datasets
    tasks_path = Path(run_config.data_folder)

    metadata = get_metadata()

    train_tasks, test_tasks = collect_files(
        tasks_path,
        metadata,
        test_config,
        run_config.is_prompting,
    )

    logger.info("Train datasets collected:")

    train_datasets = compile_datasets(
        train_tasks,
        train_config["prompt_template"],
        train_config.get("subsample_amount", None),
        train_config.get("subsample_rate", None),
        run_config.data_type
    )
    logger.info("Test datasets collected:")

    test_dataset = compile_datasets(
        test_tasks,
        test_config["prompt_template"],
        test_config.get("subsample_amount", None),
        test_config.get("subsample_rate", None),
        run_config.data_type,training = False
    )

    return train_datasets, test_dataset
