import logging
import os

from config import *
from yaml import load, Loader

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(f"/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/logs/{name}-log.log")
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