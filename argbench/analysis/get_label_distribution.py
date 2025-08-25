from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path

if __name__ == "__main__":
    args = ArgumentParser(description="Collect stats from json dataset")
    args.add_argument("-f", "--file", type=Path, required=True, help="File path")
    arg = args.parse_known_args()[0]




    with open(arg.file, "r") as f:
        dataset = json.load(f)

    print(f"Instance amount: {len(dataset['Instances'])}")

    class_counter = Counter()

    for instance in dataset["Instances"]:
        class_counter[instance["output"][0]] += 1

    print("=======================================")
    print("Class counts:")

    for i, c in enumerate(class_counter):
        count = class_counter[c]
        ratio = class_counter[c] / class_counter.total()
        if i == 0:
            print(f"| {arg.file} | {c} | {count:.3f} | {ratio:.3f} | {class_counter.total()} |")
        else:
            print(f"| {arg.file} | {c} | {count:.3f} | {ratio:.3f} |  |")
