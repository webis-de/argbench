#!/usr/bin/env python3
from common import Output, datasets_path, read_tabular, tasks_path, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import uuid

DATASET_NAME = "gretz20_conclusion_generation"


def process_dataset(dataset, path):
    output = Output(DATASET_NAME)

    output.append_definition("Judge if generated argument has a stance on topic and if its plausible. Possible responses: Plausible; Has Stance, Not Plausible; Has Stance, Plausible; Hasn't Stance, Implausible; Hasn't Stance. ")

    for row in dataset.iterrows():
        row = row[1]
        prompt = f"Topic: {row['topic']}\nArgument: {row['text']}"
        plausibility = "Plausible" if row["plausibility_label"] == "Yes" else "Not Plausible"
        stance = "Has Stance" if row["stance_label"] == "Pro" or row["stance_label"] == "Con" else "Hasn't Stance"
        model_output = f"{plausibility};{stance}"
        id = str(uuid.uuid4())
        if row["combined_label"] == 1:
            output.append_positive_example(prompt, model_output, "")
        else:
            output.append_negative_example(prompt, model_output, "")
        output.append_instance(id, prompt, [model_output])

    output.write_output(path)


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Program to convert ajjour unit segmentation dataset into appropriate form")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    dataset = str(datasets_path() / "conclusion-generation" / "gretz20-workweek-is-the-best-time-to-start-a-family" / "generated_texts_plausibility_stance.csv")

    out_path = tasks_path()

    data = read_tabular(dataset)

    metadata = Metadata(DATASET_NAME)

    process_dataset(data[data["set"] == "train"], "gretz20_conclusion_generation_train.json")
    process_dataset(data[data["set"] == "test"], "gretz20_conclusion_generation_test.json")
    process_dataset(data[data["set"] == "dev"], "gretz20_conclusion_generation_dev.json")

    metadata.add_dataset("gretz20_conclusion_generation_train.json", "train")
    metadata.add_dataset("gretz20_conclusion_generation_test.json", "test")
    metadata.add_dataset("gretz20_conclusion_generation_dev.json", "dev")

    metadata.add_evaluation_metric("f1_macro")

    metadata.write_metadata()
