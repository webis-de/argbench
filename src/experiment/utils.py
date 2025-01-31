import logging
from config import *

def get_logger(name):
    logging.basicConfig(filename="/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/logs/log.log",level=logging.INFO)
    logging.basicConfig(format="%(asctime)s %(message)s")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s %(message)s)"))
    logging.getLogger(name).addHandler(console)
    return logging.getLogger(name)