#!/usr/bin/env python3
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser(description="Converts dataset to json format")

    parser.add_argument("-f", "--file", help="Config file path")
