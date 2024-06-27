#!/usr/bin/env python3
from pathlib import Path
from common import Output, datasets_path, tasks_path, Metadata, add_seed_arg, set_seed
from dataclasses import dataclass
from argparse import ArgumentParser

DATASET_NAME = "ajjour_unit_segmentation_entity"

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
    train_path = path / "trainingSet"
    test_path = path / "testingSet"

    train_datasets = []
    for f in train_path.iterdir():
        text = extract_file(f)
        train_datasets.append(text)

    test_datasets = []
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
                    / "argument-detection"
                    / "ajjour17-unit-segmentation-of-argumentative-text/"
                    / "cross-domain"
                    / "simple")

    output_path = tasks_path()

    metadata = Metadata(DATASET_NAME)

    editorials_path = dataset_path / "editorials-split"
    essays_path = dataset_path / "essays-split"
    web_discourse_path = dataset_path / "webDiscourse-split"

    train_editorials, test_editorials = process_folder(editorials_path)
    train_editorials, test_editorials = convert_arguments(train_editorials, test_editorials)

    train_editorials.write_output("ajjour_unit_segmentation_entity_editorials_train.json")
    test_editorials.write_output("ajjour_unit_segmentation_entity_editorials_test.json")

    metadata.add_dataset("ajjour_unit_segmentation_entity_editorials_train.json", "train")
    metadata.add_dataset("ajjour_unit_segmentation_entity_editorials_test.json", "test")

    train_essays, test_essays = process_folder(essays_path)
    train_essays, test_essays = convert_arguments(train_essays, test_essays)

    train_essays.write_output("ajjour_unit_segmentation_entity_essays_train.json")
    test_essays.write_output("ajjour_unit_segmentation_entity_essays_test.json")

    metadata.add_dataset("ajjour_unit_segmentation_entity_essays_train.json", "train")
    metadata.add_dataset("ajjour_unit_segmentation_entity_essays_test.json", "test")

    train_web_discourse, test_web_discourse = process_folder(web_discourse_path)
    train_web_discourse, test_web_discourse = convert_arguments(train_web_discourse, test_web_discourse)

    train_web_discourse.write_output("ajjour_unit_segmentation_entity_web_discourse_train.json")
    test_web_discourse.write_output("ajjour_unit_segmentation_entity_web_discourse_test.json")

    metadata.add_dataset("ajjour_unit_segmentation_entity_web_discourse_train.json", "train")
    metadata.add_dataset("ajjour_unit_segmentation_entity_web_discourse_test.json", "test")

    metadata.add_evaluation_metric("rouge")

    metadata.write_metadata()
