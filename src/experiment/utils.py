import logging
from config import *

def get_logger(name):

    file_handler = logging.FileHandler(f"/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/logs/{name}.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s)"))
    logging.getLogger(name).addHandler(file_handler)
    logging.getLogger(name).setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    logging.getLogger(name).addHandler(stream_handler)
    return logging.getLogger(name)