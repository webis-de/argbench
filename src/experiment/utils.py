import logging
from config import *

def get_logger(name):

    fileHandler = logging.FileHandler(f"/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/logs/{name}.log")
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(logging.Formatter("%(asctime)s %(message)s)"))
    logging.getLogger(name).addHandler(fileHandler)
    stream_handler = logging.StreamHandler()
    logging.getLogger(name).addHandler(stream_handler)
    return logging.getLogger(name)