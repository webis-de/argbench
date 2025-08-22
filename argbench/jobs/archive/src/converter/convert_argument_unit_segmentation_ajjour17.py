#!/usr/bin/env python3
from pathlib import Path
from common import Genres, Output, Subareas, datasets_path, tasks_path, Metadata, add_seed_arg, set_seed
from dataclasses import dataclass
from argparse import ArgumentParser
from typing import List
import random

DATASET_NAME = "argument_unit_segmentation_ajjour17"

@dataclass
class TextArguments:
    fragment_id: str
    arguments: List[str]
    non_argument_seqs: List
    text: str

def extract_file(path: Path):
    datafile = open(path, "r")
    arguments = TextArguments(path.name, [], [], "")

    is_next_skip = False
    argument_idx = -1
    last_token_label = None
    non_arg_start = 0
    is_inside_non_arg = False
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
            if is_inside_non_arg:
                arguments.non_argument_seqs.append((non_arg_start, len(arguments.text)))
                is_inside_non_arg = False
            argument_idx += 1
            arguments.arguments.append("")
            arguments.arguments[argument_idx] += token
        elif token_row[0] == "Arg-I":
            arguments.arguments[argument_idx] += token
        elif token_row[0] == "Arg-O" and last_token_label == "Arg-I":
            non_arg_start = len(arguments.text)
            is_inside_non_arg = True

        arguments.text += token
        last_token_label = token_row[0]

    arguments.text.strip()
    arguments.arguments = [a.strip() for a in arguments.arguments]
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
    prompt = "Given the following document, extract all argument units in it. An argument unit is a statement that is pushed to support or attack a specific position on a topic or another statement. Argument units must be separated by newline. Do not add a new formating or enumeration. Just the argument units separated by line breaks."

    train_output = Output(DATASET_NAME)
    test_output = Output(DATASET_NAME)

    train_output.append_definition(prompt)
    test_output.append_definition(prompt)

    for dataset in train_datasets:
        extracted_arguments = "".join([f"{a}\n" for a in dataset.arguments])
        train_output.append_instance(dataset.fragment_id, dataset.text, [extracted_arguments])

    for dataset in test_datasets:
        extracted_arguments = "".join([f"{a}\n" for a in dataset.arguments])
        test_output.append_instance(dataset.fragment_id, dataset.text, [extracted_arguments])

    return train_output, test_output


def find_matching_non_arg(non_arg_seqs, bin_start, bin_end):

    matching_spans = []
    for seqs_start, seqs_end in non_arg_seqs:
        span_len = seqs_end - seqs_start
        if span_len >= bin_start:
            matching_spans.append((seqs_start, seqs_end, span_len))

    random.shuffle(matching_spans)
    matched_span = matching_spans[0]

    span_offset = int(matched_span[2] - bin_end)

    if span_offset <= 0:
        return (matched_span[0], matched_span[1])

    offset = random.randrange(span_offset)

    if random.random() < 0.5:
        return (matched_span[0] + offset, matched_span[1])
    return (matched_span[0], matched_span[1] - offset)


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

    train_editorials.write_output("argument_unit_segmentation_train_ajjour17.json")
    test_editorials.write_output("argument_unit_segmentation_test_ajjour17.json")

    metadata.add_dataset("argument_unit_segmentation_train_ajjour17.json", "train")
    metadata.add_dataset("argument_unit_segmentation_test_ajjour17.json", "test")

    metadata.add_genre(Genres.SOCIAL_MEDIA)
    metadata.add_genre(Genres.ESSAYS)
    metadata.add_subarea(Subareas.MINING)
    metadata.add_evaluation_metric("rouge")

    metadata.write_metadata()
