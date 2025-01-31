import logging


def get_logger(name):

    logging.basicConfig(format="%(asctime)s %(message)s")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s %(message)s)"))
    logging.getLogger(name).addHandler(console)
    return logging.getLogger(name)