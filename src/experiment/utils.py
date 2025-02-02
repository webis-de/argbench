import logging
from config import *

def get_logger(name):

    file_handler = logging.FileHandler(f"/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/logs/{name}-log.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s)"))
    logging.getLogger(name).addHandler(file_handler)
    logging.getLogger(name).setLevel(logging.DEBUG)
    return logging.getLogger(name)