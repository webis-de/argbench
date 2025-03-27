import json
import logging
import os
import time
from codecs import ignore_errors
import codecs
import psutil
import torch
from datetime import datetime

from debugpy.common.json import JsonObject
from pydantic import BaseModel
from yaml import load, Loader

from argbench.experiment.config import *


class Output(BaseModel):
    output: list[dict]


def get_logger(config: RunConfig):
    logger = logging.getLogger(config.get_log_path())
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(config.log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s)"))
    logger.addHandler(file_handler)

    return logger

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

def get_evaluation_metrics_map():
    evaluation_metrics = {}
    metadata = get_metadata()
    for task in metadata:
        evaluation_metrics[task] = metadata[task]["evaluation_metrics"][0]
    return evaluation_metrics

def extract_prediction(output):
    left_index = output.find("{")
    right_index = output.rindex("}")
    output_str = output[left_index:right_index+1]
    print(output_str)
    output = json.loads(output_str)
    extracted_output = output["output"]
    if type(extracted_output) == list:
        return extracted_output
    else:
        return [extracted_output]


def is_segmentation(task):
    return "segmentation" in task or "elecdeb60t020" in task or "aspect_detection" in task



def eval_collate(batch):
    out_batch = {k: [] for k in batch[0]}

    for b in batch:
        for k in b:
            out_batch[k].append(b[k])

    return out_batch

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        t = torch.cuda.mem_get_info()
        _, total_gpu = (t[0]/(1024**3),t[1]/(1024**3))
        if total_gpu < 20:
            device = torch.device("cpu")
    return device


def adjust_config(new_root_path: Path):
    """
    Modify all config files from /bigwork/nhwpajjy to new root path
    :param new_root_path: The desired root path where all code and data should reside
    :return: nothing
    """
    for file in get_config_files("  argbench/experiment/configs"):
        new_config=None
        print(file)
        with codecs.open(file, "r", encoding='utf-8-sig') as stream:
            stri = stream.read()
            config_json = json.loads(stri)
            new_config = rewrite_config(config_json, new_root_path)
        with open(file, "w", encoding='utf-8') as stream:
            json.dump(new_config, stream, indent=4)
def get_config_files(root_path: str) -> List:
    """
    Return all json files in a directory, recursively.
    :param root_path: The path where the json files resides
    :return: all json files in the directory
    """
    config_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith("json"):

                config_files.append(os.path.join(root, file))
        for subdir in dirs:
            if subdir != "archive":
                subdir_config_files = get_config_files(os.path.join(root, subdir))
                config_files.extend(subdir_config_files)
    return config_files

def rewrite_config(config: dict, new_root_path: Path):
    """
    Rewrite the paths in a dictionary so that you substitute /bigwork/nhwpajjy with the new root path
    :param config: the dictionary containing the configuration
    :param new_root_path: the new root path path
    :return:
    """
    for key in config:
        if isinstance(config[key],str):
            config[key] = config[key].replace("/bigwork/nhwpajjy",new_root_path)
        elif isinstance(config[key],dict):
            rewrite_config(config[key], new_root_path)
        elif isinstance(config[key],list):
            for obj in config[key]:
                rewrite_config(obj, new_root_path)
        else:
            pass
    return config