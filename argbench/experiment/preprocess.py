import json
import logging
import ndjson
import os
import pandas as pd
import argbench.experiment.prepare_experiment
from argparse import ArgumentParser
from pathlib import Path


from argbench.experiment.utils import *
from pathlib import Path


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
                if isinstance(instance["output"],list) or isinstance(instance["output"],str) and isinstance(instance["input"],str):
                    writer.writerow({
                        "id": instance["id"],
                        "definition": definition,
                        "input": instance["input"],
                        "output": instance["output"][0] if isinstance(instance["output"], list) else instance["output"]
                    })
                else:
                    print(instance)
    elif filetype == "parquet":
        data = []
        for instance in task_contents["Instances"]:
            data.append({
                    "id": str(instance["id"]),
                    "definition": definition,
                    "input": instance["input"],
                    "output": str(instance["output"][0]) if isinstance(instance["output"], list) else str(instance["output"])
            })
        pd.DataFrame(data).dropna().to_parquet(output_path)


multi_dataset_tasks = {"argument_unit_segmentation_ajjour17":
                           [
                            "argument_unit_segmentation_essays_ajjour17",
                            "argument_unit_segmentation_editorials_ajjour17",
                            "argument_unit_segmentation_webDiscourse_ajjour17"
                            ],
                            "controversy_scoring_cmv_adhominem_habernal18":
                            [
                                "controversy_scoring_cmv_adhominem_habernal18",
                                "reasonableness_scoring_cmv_adhominem_habernal18",

                            ],
                       }


if __name__ == "__main__":
    arg_parse = ArgumentParser(description="Convert tasks into ndjson format")
    arg_parse.add_argument("-o", "--output", required=True, type=Path, help="Output folder for processed tasks")
    arg_parse.add_argument("-f", "--filetype", default="ndjson", help="Output filetype to use")
    arg_parse.add_argument("-t", "--task",  help="specific task to prorcess")

    args = arg_parse.parse_known_args()[0]

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    tasks = []

    path = tasks_path()
    if args.task in multi_dataset_tasks:
        tasks = multi_dataset_tasks[args.task]
    else:
        if args.task:
            tasks = [args.task]
    print(f"tasks ro look for {tasks}")
    print(f"tasks path {path}")
    for item in os.listdir(path):
        print(f"iterating over {item}")
        if tasks and item not in tasks:

            continue
        task_path = path / item
        print(f"working on {task_path}")
        if not os.path.isdir(task_path):
            print(f"did not find {task_path}")
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
