#!/usr/bin/env python3
from common import Output, datasets_path, read_tabular, tasks_path, Metadata, add_seed_arg, set_seed, Genres, Skills
from argparse import ArgumentParser
import uuid
import csv
import pandas as pd

DATASET_NAME = "stance_classification_ukp_sentential_stab18"

def make_output(dataset, dataset_name):
    output = Output(DATASET_NAME)

    output.append_definition("Classify the following claim to Pro if the claim supports the topic, Con if the claim attacks the topic, or Neutral if neither. Only answer with Con, Por, or Neutral, Do not explain.")

    label_mappings = {"NoArgument": "Neutral", "Argument_for": "Pro", "Argument_against": "Con"}

    dataset["annotation"] = dataset["annotation"].apply(lambda x: label_mappings[x])

    for row in dataset.iterrows():
        row = row[1]
        prompt = f"Topic: {row['topic']}\nClaim: {row['sentence']}"
        response = row['annotation']
        id = str(uuid.uuid4())
        output.append_instance(id, prompt, [response])

    output.append_genre(Genres.WEB)
    output.append_subarea(Skills.PERSPECTIVE_ASSESSMENT)
    output.write_output(dataset_name)


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Program to convert gretz20 ibm quality dataset into appropriate form")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    metadata = Metadata(DATASET_NAME)
    dataset_path = datasets_path() / "ukp-sentential" / "data"
    datasets = []

    for dataset in dataset_path.iterdir():
        print(dataset)
        data = read_tabular(dataset, separator="\t", quoting=csv.QUOTE_NONE)
        datasets.append(data)

    datasets = pd.concat(datasets)

    #datasets["set"] = datasets["set"].map(lambda x: "train" if x == "val" or x == "train" else "test" )

    dataset_train = datasets[datasets["set"]=="train"]
    dataset_test = datasets[datasets["set"]=="test"]
    dataset_val = datasets[datasets["set"]=="val"]


    make_output(dataset_train, "stance_classification_ukp_sentential_train_stab18.json")
    make_output(dataset_test, "stance_classification_ukp_sentential_test_stab18.json")
    make_output(dataset_val, "stance_classification_ukp_sentential_val_stab18.json")

    metadata.add_dataset("stance_classification_ukp_sentential_train_stab18.json", "train")
    metadata.add_dataset("stance_classification_ukp_sentential_test_stab18.json", "test")
    metadata.add_dataset("stance_classification_ukp_sentential_val_stab18.json", "val")


    metadata.add_genre(Genres.WEB)
    metadata.add_skill(Skills.PERSPECTIVE_ASSESSMENT)
    metadata.add_evaluation_metric("fscore")
    metadata.write_metadata()
