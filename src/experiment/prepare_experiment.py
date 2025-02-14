import json
import logging
import datasets
import ndjson
import os
import pandas as pd


from collections import defaultdict
from pathlib import Path
from IPython.core.debugger import set_trace


from yaml import load, Loader
from utils import get_logger


logger = get_logger(__name__)

class dataset:
    train_path: str
    test_path: str


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

    ### Iterate over metadata task and save test and train files
    ### There are meant to be two settings
    ### 1) 1 Test dataset (fine-tuning)
    ### 2) prompting on all datasets

    test_files = {}
    if "name" in test_configs["name"]:
        test_task = test_configs["name"]
        test_files[test_task] = []
        for file in metadata[test_task]["file_list"]:
            if metadata[test_task]["split_mapping"][file] == "test":
                test_files[test_task].append(task_path / test_task / file)
                logger.info(f"adding {task_path / test_task/ file} to test")
    elif is_prompting:
        for task in metadata:
            for file in metadata[task]["file_list"]:
                if metadata[task]["split_mapping"][file] == "test":
                    logger.info(f"adding {task_path / task/ file} to test")
                    test_files[task].append(task_path / task/ file)

    train_files = defaultdict(list)
    for task in metadata:
        task_data_path = task_path / task

        if not os.path.isdir(task_data_path):
            logger.warning(f"{task} not found!")
            continue


        logger.debug(f"reading {task_data_path}")

        for task_file in os.listdir(task_data_path):
            task_file_path = task_data_path / task_file
            logger.info(f"adding {task_file_path} to training")

            if not os.path.isfile(task_file_path) or task_file_path in test_files:
                continue
            if metadata[task]["split_mapping"][task_file] == "train":
                train_files[task].append(task_file_path)
    return train_files, test_files


def compile_datasets(
        train_files,
        test_files,
        prompt_template,
        test_subsample_amount=None,
        test_subsample_rate=None,
        train_subsample_amount=None,
        train_subsample_rate=None,
        is_prompt=False, test_dataset=None):
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

    test_datasets = {}
    training_datasets = {}
    for dataset in train_files:
        if not is_prompt or train_subsample_amount:
            train_path = train_files[dataset]
            df_training = pd.read_json(train_path, lines=True)
            if train_subsample_amount :
                df_training= df_training.sample(train_subsample_amount)
            elif train_subsample_rate:
                df_training= df_training.sample(frac=train_subsample_rate)
            for column in df_training.columns:
                df_training[column] = df_training[column].astype(str)

            df_training = df_training[["id","document", "input", "output", "task"]]
            df_training["task"] = dataset
            if not is_prompt:
                training_datasets[dataset]=df_training

        test_path = test_files[dataset]
        if test_dataset and dataset !=test_dataset:
            continue
        df_test = pd.read_json(test_path, lines=True)
        if test_subsample_rate:
            df_test = df_test.sample(frac=test_subsample_rate)
        elif test_subsample_amount:
            df_test = df_test.sample(test_subsample_amount)
        if is_prompt and train_subsample_amount:
            example_str = ""
            for _, example in df_training.iterrows():
                example_instance = f'Input: {example["input"].values[0]}\nOutput: {example["output"].values[0]}'
                example_str += example_instance

            df_test.rename(columns={"input": "document"}, inplace=True)
            df_test["example"] = example_str
            df_test["input"] = df_test.apply(template_formatter, axis=1)

        for column in df_test.columns:
            df_test[column] = df_test[column].astype(str)
        df_test["task"] = dataset
        df_test = df_test[["id","document", "input", "output", "task"]]
        test_datasets[dataset]= df_test


#



    return training_datasets, test_datasets


def collect_datasets(run_config):
    """
    Use RunConfig to create train and test datasets

    :param run_config: RunConfig with train_datasets and test_datasets config dicts
    :returns: Tuple of train and test datasets in pandas DataFrame
    """
    train_config = run_config.train_datasets
    test_config = run_config.test_dataset
    tasks_path = Path(run_config.data_folder)
    prompt_template = run_config.model_config.prompt_template
    metadata = get_metadata()

    train_files, test_files = collect_files(
        tasks_path,
        metadata,
        test_config,
        run_config.is_prompting,
    )

    logger.info("Train datasets collected:")



    train_datasets, test_datasets  = compile_datasets(
        train_files,
        test_files,
        prompt_template,
        test_config.get("subsample_amount", None),
        test_config.get("subsample_rate", None),
        train_config.get("subsample_amount", None),
        train_config.get("subsample_rate", None)
        ,run_config.is_prompting
    )
    if run_config.is_prompting:
        return train_datasets, test_datasets
    else:
        return train_datasets, test_datasets[run_config.test_dataset.name]
