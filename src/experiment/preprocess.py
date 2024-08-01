from argparse import ArgumentParser
import json
from pathlib import Path
import random
from string import Formatter
import re
from yaml import load, Loader
import random
import ndjson
import os

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

def compile_template_vars(dataset, template):
    """Compiles prompt based on config template"""
    template_variables = [var for _, var, _, _ in Formatter().parse(template) if var]

    positive_amount = sum("positive_example_input" in var for var in template_variables)
    negative_amount = sum("negative_example_input" in var for var in template_variables)

    template_values = {
        "definition": dataset["Definition"][0],
    }

    if dataset["Positive Examples"]:
        positive_chosen = random.sample(dataset["Positive Examples"], k=positive_amount)
    else:
        positive_chosen = []
    if dataset["Negative Examples"]:
        negative_chosen = random.sample(dataset["Negative Examples"], k=negative_amount)
    else:
        negative_chosen = []

    for i in range(positive_amount):
        template_input_var = f"positive_example_input_{i}"
        template_output_var = f"positive_example_output_{i}"
        template_values[template_input_var] = positive_chosen[i]["input"]
        template_values[template_output_var] = positive_chosen[i]["output"]

    for i in range(negative_amount):
        template_input_var = f"negative_example_input_{i}"
        template_output_var = f"negative_example_output_{i}"
        template_values[template_input_var] = negative_chosen[i]["input"]
        template_values[template_output_var] = negative_chosen[i]["output"]

    return template_values

def compile_instances(dataset, template, instances):
    """Compile all dataset instances to one array"""
    template_vars = compile_template_vars(dataset, template)

    for instance in dataset["Instances"]:
        prompt = template.format(
            instance_input=instance["input"],
            **template_vars
        )
        try:
            output = instance["output"][0]
        except IndexError:
            print(template)
            print(dataset["Instances"][0])
            raise Exception()

        instances.append({
            "id": instance["id"],
            "input": prompt,
            "output": output
        })


def append_dataset(datasets, dataset_config, dataset_file_path, data_instances):
    for dataset in datasets["file_list"]:
        if not re.match(dataset_config["match"], dataset):
            continue
        with open(dataset_file_path / dataset, "r") as f:
            dataset_contents = json.load(f)
        compile_instances(dataset_contents, dataset_config["prompt_template"], data_instances)


def collect_datasets(config, metadata, datasets_path):
    """Collect used datasets"""
    train_instances = []
    test_instances = []

    for i, dataset in enumerate(config["train_datasets"]):
        dataset_files = metadata[dataset]
        dataset_config = config["train_datasets"][dataset]
        dataset_file_path = datasets_path / dataset
        append_dataset(dataset_files, dataset_config, dataset_file_path, train_instances)
        if dataset_config.get("subsample_rate"):
            data_indices = [n for n in range(len(train_instances[i]))]
            data_indices = random.sample(data_indices, int(len(data_indices) * dataset_config["subsample_rate"]))
            train_instances[i] = [train_instances[n] for n in data_indices]

    for i, dataset in enumerate(config["test_datasets"]):
        dataset_files = metadata[dataset]
        dataset_config = config["test_datasets"][dataset]
        dataset_file_path = datasets_path / dataset
        append_dataset(dataset_files, dataset_config, dataset_file_path, test_instances)
        if dataset_config.get("subsample_rate"):
            data_indices = [i for i in range(len(test_instances[i]))]
            data_indices = random.sample(data_indices, int(len(data_indices) * dataset_config["subsample_rate"]))
            test_instances[i] = [test_instances[n] for n in data_indices]

    random.shuffle(train_instances)
    random.shuffle(test_instances)

    return train_instances, test_instances

def validate_config(config):
    """Validate configuration file"""
    assert config.get("seed")
    assert config.get("dataset_output")
    assert all(split in ["train", "test"] for split in config["dataset_output"])
    assert len(config["train_datasets"]) > 0
    for dataset in config["train_datasets"]:
        dataset_config = config["train_datasets"][dataset]
        assert dataset_config.get("match")
        assert dataset_config.get("prompt_template")

    assert len(config["test_datasets"]) > 0
    for dataset in config["test_datasets"]:
        dataset_config = config["test_datasets"][dataset]
        assert dataset_config.get("match")
        assert dataset_config.get("prompt_template")

if __name__ == "__main__":
    argument_parser = ArgumentParser(description="Compile dataset to use")
    argument_parser.add_argument("-c", "--config", type=Path, required=True, help="Path to the config file to use")
    args = argument_parser.parse_known_args()[0]

    with open(args.config, "r") as f:
        config = load(f, Loader=Loader)

    validate_config(config)

    datasets_path = tasks_path()
    metadata = get_metadata()

    train_instances, test_instances = collect_datasets(config, metadata, datasets_path)

    with open(config["dataset_output"]["train"], "w") as f:
        ndjson.dump(train_instances, f)

    with open(config["dataset_output"]["test"], "w") as f:
        ndjson.dump(test_instances, f)
