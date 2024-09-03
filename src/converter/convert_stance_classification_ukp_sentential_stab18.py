#!/usr/bin/env python3
from common import Output, datasets_path, read_tabular, tasks_path, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import uuid

DATASET_NAME = "stance_classification_ukp_sentential_stab18"

def make_output(dataset, dataset_name):
    output = Output(DATASET_NAME)

    output.append_definition("Classify the following claim to Pro if the claim supports the topic, Con if the claim attacks the topic, or Neutral if neither.")

    label_mappings = {"NoArgument": "Neutral", "Argument_for": "Pro", "Argument_against": "Con"}

    dataset["annotation"] = dataset["annotation"].apply(lambda x: label_mappings[x])

    for row in dataset.iterrows():
        row = row[1]
        prompt = f"Topic: {row['topic']}\nClaim: {row['sentence']}"
        response = row['annotation']
        id = str(uuid.uuid4())
        output.append_instance(id, prompt, [response])

    output.write_output(dataset_name)


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Program to convert gretz20 ibm quality dataset into appropriate form")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    metadata = Metadata(DATASET_NAME)
    dataset_path= str(datasets_path() / "ukp-sentential/corpus-ukp-sentential-argument-mining.csv")


    dataset = read_tabular(dataset_path)
    dataset_train = dataset[dataset["set"]=="train"]
    dataset_test = dataset[dataset["set"]=="test"]
    dataset_val = dataset[dataset["set"]=="val"]

    make_output(dataset_train, "stance_classification_ukp_sentential_train_stab18.json")
    make_output(dataset_test, "stance_classification_ukp_sentential_test_stab18.json")
    make_output(dataset_val, "stance_classification_ukp_sentential_val_stab18.json")

    metadata.add_dataset("stance_classification_ukp_sentential_train_stab18.json", "train")
    metadata.add_dataset("stance_classification_ukp_sentential_test_stab18.json", "test")
    metadata.add_dataset("stance_classification_ukp_sentential_val_stab18.json", "dev")
    metadata.add_evaluation_metric("f1_macro")
    metadata.write_metadata()
