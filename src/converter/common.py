import random
from typing import List
import pandas as pd
import json
from pathlib import Path
import os
import numpy as np
from yaml import load, Loader

def read_tabular(path, separator=","):
    """Reads tabular csv file"""
    dataset = pd.read_csv(path, sep=separator)
    return dataset

def data_repo():
    """Get path of data repository"""
    curr_file = os.path.abspath(__file__)
    config_file = Path(curr_file).parents[2] / "config.yaml"
    with open(config_file, "r") as f:
        config = load(f, Loader=Loader)
        return Path(config["data_repo"])

def datasets_path():
    """Get path to dataset folder"""
    dataset_folder = data_repo() / "datasets"
    return dataset_folder

def tasks_path():
    """Path to tasks folder"""
    dataset_folder = data_repo() / "tasks"
    return dataset_folder

def add_seed_arg(argparser):
    """Adds seed argument to argument parser"""
    argparser.add_argument("-s", "--seed", default=42, help="Seed to use for dataset generation")

def set_seed(parsed_args):
    random.seed(parsed_args.seed)
    np.random.seed(parsed_args.seed)


class Output:
    """Output object"""

    def __init__(self, dataset_name) -> None:
        self.dataset_name = dataset_name
        self.tasks_path = tasks_path() / self.dataset_name
        self.contributors = []
        self.source = []
        self.url = []
        self.categories = []
        self.reasoning = []
        self.definition = []
        self.input_language = []
        self.output_language = []
        self.instruction_language = []
        self.domains = []
        self.positive_examples = []
        self.negative_examples = []
        self.instances = []
        self.genre = []
        self.instance_license = []


    def append_positive_example(self, input: str, output: str, explanation: str):
        self.positive_examples.append({"input": input, "output": output, "explanation": explanation})


    def append_negative_example(self, input: str, output: str, explanation: str):
        self.negative_examples.append({"input": input, "output": output, "explanation": explanation})


    def append_instance(self, id: str, input: str, output: List[str]):
        self.instances.append({"id": id, "input": input, "output": output})


    def append_definition(self, definition: str):
        self.definition.append(definition)


    def append_subarea(self, subarea: str):
        self.categories.append(subarea)


    def append_genre(self, genre: str):
        self.genre.append(genre)


    def write_output(self, file_name):
        output = {
            "Contributors": self.contributors,
            "Source": self.source,
            "URL": self.url,
            "Categories": self.categories,
            "Reasoning": self.reasoning,
            "Definition": self.definition,
            "Input_language": self.input_language,
            "Output_language": self.output_language,
            "Instruction_language": self.instruction_language,
            "Domains": self.domains,
            "Positive Examples": self.positive_examples,
            "Negative Examples": self.negative_examples,
            "Instances": self.instances,
            "Genre": self.genre,
            "Instance License": self.instance_license
        }

        self.tasks_path.mkdir(parents=True, exist_ok=True)

        with open(self.tasks_path / file_name, "w+") as f:
            json.dump(output, f, indent=2)

class Metadata:
    """Metadata object for the dataset"""

    metadata_path = tasks_path() / "metadata.json"
    evaluation_metrics = [
        "f1_macro",
        "f1_micro",
        "rouge",
        "kendalltau"
    ]

    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name
        if os.path.isfile(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                self.dataset_data = json.load(f)
        else:
            self.dataset_data = {}

        self.dataset_data[self.dataset_name] = {
            "file_list": [],
            "split_mapping": {},
            "evaluation_metrics": []
        }


    def add_dataset(self, dataset_file, dataset_split=None):
        """Add Dataset to metadata"""
        dataset_split = dataset_split if dataset_split else "none"
        self.dataset_data[self.dataset_name]["file_list"].append(dataset_file)
        self.dataset_data[self.dataset_name]["split_mapping"][dataset_file] = dataset_split


    def add_evaluation_metric(self, metric):
        """Add metric to evaluate dataset on"""
        assert metric in self.evaluation_metrics, f"Metric should be one of: {self.evaluation_metrics}"
        self.dataset_data[self.dataset_name]["evaluation_metrics"].append(metric)


    def write_metadata(self):
        """Write metadata file to disk"""
        with open(self.metadata_path, "w") as f:
            json.dump(self.dataset_data, f, indent=2)
