import json
import logging
import ndjson
import os
import pandas as pd

from argparse import ArgumentParser
from pathlib import Path
from preprocess import tasks_path

logger = logging.getLogger(__name__)


def process_task_file(output_path, task_file_path, filetype="ndjson"):
    """
    Read task file in .json format and process it into ndjson format

    :param output_path: Path to output file
    :param task_file_path: Original task file path
    """
    with open(task_file_path, "r") as f:
        task_contents = json.load(f)

    if not (
            isinstance(task_contents, dict) and
            task_contents.get("Definition") and
            task_contents.get("Instances")):
        print(f"Skipped Malformed file: {task_file_path}")
        return

    definition = task_contents["Definition"][0] if isinstance(task_contents["Definition"], list) else task_contents["Definition"]

    if filetype == "ndjson":
        with open(output_path, "w") as f:
            writer = ndjson.writer(f, ensure_ascii=False)
            for instance in task_contents["Instances"]:
                writer.writerow({
                    "id": instance["id"],
                    "definition": definition,
                    "input": instance["input"],
                    "output": instance["output"][0] if isinstance(instance["output"], list) else instance["output"]
                })
    elif filetype == "parquet":
        data = []
        for instance in task_contents["Instances"]:
            data.append({
                    "id": str(instance["id"]),
                    "definition": definition,
                    "input": instance["input"],
                    "output": str(instance["output"][0]) if isinstance(instance["output"], list) else str(instance["output"])
            })
        pd.DataFrame(data).to_parquet(output_path)



if __name__ == "__main__":
    arg_parse = ArgumentParser(description="Convert tasks into ndjson format")
    arg_parse.add_argument("-o", "--output", required=True, type=Path, help="Output folder for processed tasks")
    arg_parse.add_argument("-f", "--filetype", default="ndjson", help="Output filetype to use")
    arg_parse.add_argument("-t", "--task", default="ndjson", help="specific task to prorcess")

    args = arg_parse.parse_known_args()[0]

    if not os.path.exists(args.output):
        os.mkdir(args.output)


    path = tasks_path()

    for item in os.listdir(path):
        if args.task and item!=args.task:
            continue
        task_path = path / item
        if not os.path.isdir(task_path):
            continue

        if not os.path.exists(args.output / item):
            os.mkdir(args.output / item)

        print("========================")
        print(f"Processing task: {item}")
        print("========================")

        for task_item in os.listdir(task_path):
            task_item_path = task_path / task_item
            if not os.path.isfile(task_item_path) or ".json" not in task_item:
                continue

            print(f"Processing file: {task_item}")
            print("-----------------------------")

            process_task_file(
                args.output / item / task_item,
                task_item_path,
                args.filetype
            )
