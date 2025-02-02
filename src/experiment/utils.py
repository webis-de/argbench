import logging
from config import *

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(f"/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/logs/{name}-log.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s)"))
    logger.addHandler(file_handler)

    return logger