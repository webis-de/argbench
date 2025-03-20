import logging
import os
import time

import psutil
import torch
from datetime import datetime
from pydantic import BaseModel
from yaml import load, Loader

from argbench.experiment.config import *


class Output(BaseModel):
    output: list[dict]


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    now = datetime.now()
    starting_time = now.strftime("%m-%d-%H:%M:%S")
    file_handler = logging.FileHandler(f"/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/logs/{name}-{starting_time}log.log")
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

