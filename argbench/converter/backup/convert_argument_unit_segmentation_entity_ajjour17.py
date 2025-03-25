#!/usr/bin/env python3
from pathlib import Path
from common import Output, datasets_path, tasks_path, Metadata, add_seed_arg, set_seed, Genres, Subareas
from dataclasses import dataclass
from argparse import ArgumentParser

DATASET_NAME = "argument_unit_segmentation_entity_ajjour17"

@dataclass
class TextArguments:
    fragment_id: str
    text: str
    output_text: str

def extract_file(path: Path):
    datafile = open(path, "r")
    arguments = TextArguments(path.name, "", "")

    is_next_skip = False
    argument_idx = -1
    for line in datafile:

        token_row = line.split("\t\t")

        if token_row[5] == "POS" and token_row[1] == "'":
            token = token_row[1]
            is_next_skip = True
        elif token_row[5] == "(":
            token = " " + token_row[1]
            is_next_skip = True
        elif (token_row[5] == "SENT"
              or token_row[5] == ","
              or token_row[5] == "''"
              or token_row[5] == ")"
              or token_row[5] == ":"):
            token = token_row[1]
        elif is_next_skip:
            token = token_row[1]
            is_next_skip = False
        else:
            token = " " + token_row[1]

        if token_row[0] == "Arg-B":
            argument_idx += 1
            arguments.output_text
            arguments.output_text += " Begin-argument"
        elif token_row[0] == "Arg-I" and not is_next_skip:
            arguments.output_text += " argument"
        else:
            arguments.output_text += token

        arguments.text += token

    arguments.text.strip()
    arguments.output_text.strip()
    datafile.close()
    return arguments


def process_folder(path: Path):
    train_datasets = []
    test_datasets = []
    for split in path.iterdir():
        train_path = split / "trainingSet"
        test_path = split / "testingSet"

        for f in train_path.iterdir():
            text = extract_file(f)
            train_datasets.append(text)

        for f in test_path.iterdir():
            text = extract_file(f)
            test_datasets.append(text)

    return train_datasets, test_datasets


def convert_arguments(train_datasets, test_datasets):
    prompt = "Given the following document, extract all argument units in it by labeling the begining of an argument unit as Begin-argument and the each other token of the argument unit as argument. An argument unit is a statement that is pushed to support or attack a specific position on a topic or another statement."

    train_output = Output(DATASET_NAME)
    test_output = Output(DATASET_NAME)

    train_output.append_definition(prompt)
    test_output.append_definition(prompt)

    for dataset in train_datasets:
        train_output.append_instance(dataset.fragment_id, dataset.text, [dataset.output_text])

    for dataset in test_datasets:
        test_output.append_instance(dataset.fragment_id, dataset.text, [dataset.output_text])

    return train_output, test_output


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Program to convert ajjour unit segmentation dataset into appropriate form")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    dataset_path = (datasets_path()
                    / "web-discourse"
                    / "segmentation-splits"
                    / "simple")

    output_path = tasks_path()

    metadata = Metadata(DATASET_NAME)

    train_editorials, test_editorials = process_folder(dataset_path)
    train_editorials, test_editorials = convert_arguments(train_editorials, test_editorials)

    train_editorials.append_genre(Genres.ESSAYS)
    train_editorials.append_genre(Genres.SOCIAL_MEDIA)
    train_editorials.append_subarea(Subareas.MINING)
    test_editorials.append_genre(Genres.ESSAYS)
    test_editorials.append_subarea(Subareas.MINING)

    train_editorials.write_output("argument_unit_segmentation_entity_train_ajjour17.json")
    test_editorials.write_output("argument_unit_segmentation_entity_test_ajjour17.json")

    metadata.add_dataset("argument_unit_segmentation_entity_train_ajjour17.json", "train")
    metadata.add_dataset("argument_unit_segmentation_entity_test_ajjour17.json", "test")

    metadata.add_genre(Genres.SOCIAL_MEDIA)
    metadata.add_genre(Genres.ESSAYS)
    metadata.add_skill(Subareas.MINING)
    

    metadata.write_metadata()
