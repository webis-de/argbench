#!/usr/bin/env python3
from common import Output, datasets_path, read_tabular, tasks_path, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import uuid

DATASET_NAME = "stance_classification_ukp_sentential_stab18"

def make_output(dataset, dataset_name):
    output = Output(DATASET_NAME)

    output.append_definition("Identify sentiment of claim towards topic. Possible outputs: Pro if claim supports topic or Con if claim does not support claim towards target.")

    for row in dataset.iterrows():
        row = row[1]
        prompt = f"Topic: {row['topic']}\nClaim: {row['text']}"
        response = "Pro" if row["label"] == 1 else "Con"
        wrong_response = "Pro" if row["label"] == 0 else "Con"
        id = str(uuid.uuid4())

        output.append_positive_example(prompt, response, "")
        output.append_negative_example(prompt, wrong_response, "")

        output.append_instance(id, prompt, [response])

    output.write_output(dataset_name)


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Program to convert gretz20 ibm quality dataset into appropriate form")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    metadata = Metadata(DATASET_NAME)

    dataset_train_path = str(datasets_path()
                    / "ukp-sentential"
                    / "argmin_train.csv")

    dataset_test_path = str(datasets_path()
                    / "ukp-sentential"
                    / "argmin_test.csv")

    dataset_val_path = str(datasets_path()
                    / "ukp-sentential"
                    / "argmin_val.csv")

    dataset_train = read_tabular(dataset_train_path)
    dataset_test = read_tabular(dataset_test_path)
    dataset_val = read_tabular(dataset_val_path)

    make_output(dataset_train, "stance_classification_ukp_sentential_train_stab18.json")
    make_output(dataset_test, "stance_classification_ukp_sentential_test_stab18.json")
    make_output(dataset_val, "stance_classification_ukp_sentential_val_stab18.json")

    metadata.add_dataset("stance_classification_ukp_sentential_train_stab18.json", "train")
    metadata.add_dataset("stance_classification_ukp_sentential_test_stab18.json", "test")
    metadata.add_dataset("stance_classification_ukp_sentential_val_stab18.json", "dev")
    metadata.add_evaluation_metric("f1_macro")
    metadata.write_metadata()
