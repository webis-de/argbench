import json
from collections import defaultdict
from typing import Dict, Set
import logging
import numpy as np

from argbench.converter.common import *
from argbench.experiment.preprocess import *


class ExperimentType(Enum):
    IN_TASK = "in_task"
    CROSS_TASK = "cross_task"

class DatasetSplit(Enum):
    TRAIN = "train"
    TEST = "test"
    VAL = "val"
    TRAIN_AND_VAL = "train_and_val"

logger = logging.getLogger()



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




def get_dataset_split(dataset, set, metadata, task_data_path):
    for file in metadata[dataset]["file_list"]:
        if metadata[dataset]["split_mapping"][file] == set:

            return task_data_path / dataset / file
    return None

def load_set(dataset, task_data_path, split, sample_rate:float = None, sample_size: int = None):
    metadata = get_metadata()
    print(dataset)
    if split == DatasetSplit.TRAIN_AND_VAL:
        train_set,_ = load_set(dataset, task_data_path, DatasetSplit.TRAIN, sample_rate, sample_size)
        test_set,_ = load_set(dataset, task_data_path, DatasetSplit.VAL, sample_rate, sample_size)
        return pd.concat([train_set, test_set]), None
    split = split.value
    path = get_dataset_split(dataset, split, metadata, task_data_path)
    if sample_rate or sample_size:
        path_sample = formulate_sample_path(path, sample_rate, sample_size)
    else:
        path_sample = path
    if os.path.exists(path_sample):
        return  pd.read_json(path_sample, lines=True), path_sample
    else:
        return sample_set(path, sample_rate, sample_size)

def formulate_sample_path(output_path: Path, sample_rate: float = None, sample_size: int= None):
    base, file_name = os.path.split(output_path)
    if sample_rate:
        sample_name = file_name.replace(".json", f"-rate-{sample_rate}.json")
    elif sample_size:
        sample_name = file_name.replace(".json", f"-size-{sample_size}.json")
    else:
        raise ValueError("no rate or size defined")

    return os.path.join(base, sample_name)

def sample_set(output_path: Path, sample_rate: float = None, sample_size: int = None ):

    df_set = pd.read_json(output_path, lines=True)
    path_sample = formulate_sample_path(output_path, sample_rate, sample_size)
    if sample_rate:
        df_sample = df_set.sample(frac=sample_rate)
    elif sample_size:
        df_sample = df_set.sample(sample_size)
    else:
        raise ValueError("no rate or size defined")
    df_sample.to_json(path_sample, orient='records', lines=True)
    return df_sample, path_sample


def split_datasets_fine_tuning(task_data_path,
                               prompt_template,
                               test_subsample_amount,
                               test_subsample_rate,
                               train_subsample_amount,
                               train_subsample_rate,
                               test_dataset, experiment_type, experiment_splits, is_validate=False, task_filter=None):

    def template_formatter(row):
        return prompt_template.format(instance_input=row["document"], definition=row["definition"])

    if experiment_type == ExperimentType.IN_TASK:
        experiment_split_training = {test_dataset}
        experiment_split_test = {test_dataset}
        logger.info(f"In-task experiment on {test_dataset}")
    else:
        experiment_split_test, experiment_split_training = get_experiment_split(is_validate, experiment_splits)
        logger.info(f"Cross-task experiment")
        logger.info(f"Cross-task experiment")

    if not test_dataset :
        raise ValueError("test dataset must be set")

    if test_dataset not in experiment_split_test:
        logger.warning(f"{test_dataset} not in experiment split_test {experiment_split_test}")

    if is_validate:
        train_dataset_split = DatasetSplit.TRAIN
        test_dataset_split = DatasetSplit.VAL
        logger.info(f"Validation experiment")
    else:
        train_dataset_split = DatasetSplit.TRAIN_AND_VAL
        test_dataset_split = DatasetSplit.TEST
        logger.info(f"Training experiment")

    test_datasets = {}
    training_datasets = {}

    if task_filter:
        experiment_split_training = set(experiment_split_training).intersection(task_filter)


    for dataset in experiment_split_training:
        df_training, train_path = load_set(dataset, task_data_path, train_dataset_split, sample_size=train_subsample_amount, sample_rate=train_subsample_rate)
        for column in df_training.columns:
            df_training[column] = df_training[column].astype(str)
        df_training.path = train_path
        df_training = df_training[["id", "input", "output"]]
        training_datasets[dataset] = df_training

    for dataset in experiment_split_test:
        df_test, test_path = load_set(dataset, task_data_path, test_dataset_split, sample_rate=test_subsample_rate, sample_size=test_subsample_amount)
        df_test.path = test_path
        df_test.rename(columns={"input": "document"}, inplace=True)
        df_test["input"] = df_test.apply(template_formatter, axis=1)
        df_test = df_test[["id","document", "input", "output"]]
        test_datasets[dataset]= df_test

    return training_datasets, test_datasets

def split_datasets_prompting(
        task_data_path,
        prompt_template,
        test_subsample_amount,
        test_subsample_rate,
        train_subsample_amount,
        train_subsample_rate,
        test_dataset=None):
    """
    Read dataset file and compile all datasets into one dataframe

    :param task_datasets: List of dataset file paths for each dataset
    :param prompt_template: Template to compile dataset variables into prompt
    :param subsample_amount: Amount of samples to take from dataset file
    :param subsample_rate: % of instances to take from dataset
    :returns: Full compiled DataFrame of all datasets instances
    """
    def few_shot_template_formatter(row):
        return prompt_template.format(instance_input=row["document"], definition=row["definition"], example=row["example"])

    def template_formatter(row):
        return prompt_template.format(instance_input=row["document"], definition=row["definition"])

    test_datasets = {}
    training_datasets = {}
    tasks = get_metadata()

    for dataset in tasks:
        if test_dataset and dataset !=test_dataset:
            continue

        df_training, train_path = load_set(dataset, task_data_path, DatasetSplit.TRAIN_AND_VAL, sample_size=train_subsample_amount,
                                               sample_rate = train_subsample_rate)
        df_training.path = train_path
        for column in df_training.columns:
            df_training[column] = df_training[column].astype(str)
        df_training = df_training[["id", "input", "output"]]



#        df_test['output'] = df_test['output'].apply(json.dumps)
        df_test, test_path = load_set(dataset, task_data_path, DatasetSplit.TEST, sample_size=test_subsample_amount,
                                          sample_rate = test_subsample_rate)

        df_test.path = test_path

        ###  Formatting
        if train_subsample_amount:
            example_str = ""
            for _, example in df_training.iterrows():
                example_instance = f'Input: {example["input"]}\nOutput: {example["output"]}'
                example_str += example_instance
            df_test.rename(columns={"input": "document"}, inplace=True)
            df_test["example"] = example_str
            df_test["input"] = df_test.apply(few_shot_template_formatter, axis=1)
        else:
            df_test.rename(columns={"input": "document"}, inplace=True)
            df_test["input"] = df_test.apply(template_formatter, axis=1)

        for column in df_test.columns:
            df_test[column] = df_test[column].astype(str)

        df_test = df_test[["id","document", "input", "output"]]
        test_datasets[dataset]= df_test

    return training_datasets, test_datasets


def collect_datasets(run_config):
    """
    Use RunConfig to create train and test datasets

    :param run_config: RunConfig with train_datasets and test_datasets config dicts
    :returns: Tuple of train and test datasets in pandas DataFrame
    """
    global logger
    logger = get_logger(run_config)
    train_config = run_config.train_datasets
    test_config = run_config.test_dataset
    tasks_path = Path(run_config.data_folder)
    if run_config.is_prompting and run_config.is_chain_of_thoughts:
        prompt_template = run_config.model_config.cot_prompt_template
    else:
        prompt_template = run_config.model_config.prompt_template




    logger.info("Train datasets collected:")
    with open(run_config.experiment_splits_path) as experiment_splits_file:
        experiment_splits = json.load(experiment_splits_file)

    if run_config.skill_filter:
        task_filter = get_filters_by_skill(run_config.skill_filter)
    else:
        task_filter = {}

    test_subsample_amount = test_config.get("subsample_amount", None)
    test_subsample_rate = test_config.get("subsample_rate", None)
    train_subsample_amount = train_config.get("subsample_amount", None)
    train_subsample_rate = train_config.get("subsample_rate", None)
    test_dataset = test_config.get("name", None)
    is_validate = run_config.is_hpo
    if run_config.is_in_task:
        experiment_type = ExperimentType.IN_TASK
    else:
        experiment_type = ExperimentType.CROSS_TASK

    if run_config.is_prompting:
        train_datasets, test_datasets  = split_datasets_prompting(tasks_path, prompt_template,
            test_subsample_amount, test_subsample_rate, train_subsample_amount, train_subsample_rate, test_dataset)
    else:
        train_datasets, test_datasets  = split_datasets_fine_tuning(tasks_path, prompt_template,
        test_subsample_amount, test_subsample_rate, train_subsample_amount, train_subsample_rate, test_dataset,
                                    experiment_type, experiment_splits, is_validate, task_filter)

    return train_datasets, test_datasets


def get_filters_by_skill(skill: str) -> Dict[str, float]:
    filter = {}
    metadata = get_metadata()
    for task in metadata:
        if metadata[task]["skill"] == skill:
            filter[task] = 1
    return filter


def get_experiment_split(is_validate, experiment_splits) -> (Set[str], Set[str]):
    """

    :param is_validate: whether the experiment is a validation or test experiment
    :param experiment_splits: a dictionary containing
    :return:
    """
    if is_validate:
        test = experiment_splits["validation"]
        training = experiment_splits["training"]
    else:
        test = experiment_splits["test"]
        training = experiment_splits["validation"] + experiment_splits["training"]
    return test, training


def save_experiment_splits(experiment_splits_path):
    """
    This function randomly choose five tasks for validation, five for test, and the rest for training
    :param run_config: the main configuration object
    :return: None
    """


    metadata = get_metadata()
    tasks_per_skill = defaultdict(list)
    experiment_splits = defaultdict(list)

    for task in metadata:
        skill = metadata[task]["skill"]
        tasks_per_skill[skill].append(task)

    for skill in tasks_per_skill:
        val_task = np.random.choice(tasks_per_skill[skill])
        test_task = np.random.choice(tasks_per_skill[skill])
        while test_task == val_task:
            test_task = np.random.choice(tasks_per_skill[skill])

        experiment_splits["validation"].append(val_task)
        experiment_splits["test"].append(test_task)

    for task in metadata:
        if task not in experiment_splits["test"] and task not in experiment_splits["validation"]:
            experiment_splits["training"].append(task)

    with open(experiment_splits_path, "w") as file_stream:
        json.dump(experiment_splits, file_stream, indent=4, sort_keys=True)






