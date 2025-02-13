#!/usr/bin/env python3
from pathlib import Path
from common import Genres, Output, Skills, datasets_path, tasks_path, Metadata, add_seed_arg, set_seed
from dataclasses import dataclass
from argparse import ArgumentParser
from typing import List
from tqdm import tqdm
import random



@dataclass
class Unit:
    span: str
    label: str

@dataclass
class Units:
    fragment_id: str
    units: List #( span, argumentative or non-argumentative)
    
    text: str

def extract_file(path: Path):
    datafile = open(path, "r")
    units = Units(path.name, [], "")

    is_next_skip = False
    idx = -1
    last_token_label = None
    non_arg_start = 0
    is_inside_non_arg = False
    for i, line in tqdm(enumerate(datafile)):

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

                is_inside_non_arg = False
            idx += 1
            units.units.append(Unit("", "Argumentative"))
            units.units[idx].span += token

        elif token_row[0] == "Arg-I":
            units.units[idx].span += token

        elif token_row[0] == "Arg-O" and (last_token_label == "Arg-I" or i == 0):
            is_inside_non_arg = True
            idx += 1
            units.units.append(Unit("", "Non-argumentative"))
            units.units[idx].span += token

        elif token_row[0] == "Arg-O" and is_inside_non_arg:
            units.units[idx].span += token


        units.text += token
        last_token_label = token_row[0]

    units.text.strip()

    datafile.close()
    return units

DATASET_NAME_TEMPLATE = "argument_unit_segmentation_{dataset}_ajjour17"
DATASET_FILE_TEMPLATE = "argument_unit_segmentation_{dataset}_{split}_ajjour17"

def process_folder(path: Path):
    train_dataset = []
    test_dataset = []
    train_path = path / "trainingSet"
    test_path = path / "testingSet"

    for f in train_path.iterdir():
        text = extract_file(f)
        train_dataset.append(text)

    for f in test_path.iterdir():
        text = extract_file(f)
        test_dataset.append(text)

    return train_dataset, test_dataset


def convert_arguments(dataset_name, test_dataset, train_dataset):
    prompt = """Given the following document, split all of the document into argumentative units and non-argumentative units.
An argumentative unit is a statement that has an argumentative function for example a claim or anecdote.
Prepend each argumentative unit with argumentative: and spans that are not Argumentative with Non-argumentative:.
Output the extracted spans as they are ordered in the given document and separate them by a new line.
Do not add a new formating or enumeration also do not rephrase the argument units. Order the output spans as they appear in the document."""

    train_output = Output(dataset_name)
    test_output = Output(dataset_name)

    train_output.append_definition(prompt)
    test_output.append_definition(prompt)

    for file in tqdm(train_dataset):
        extracted_units = "".join([f"{unit.label}: {unit.span.strip()}\n" for unit in file.units])
        train_output.append_instance(file.fragment_id, file.text, [extracted_units])

    for file in tqdm(test_dataset):
        extracted_units = "".join([f"{unit.label}: {unit.span.strip()}\n" for unit in file.units])
        test_output.append_instance(file.fragment_id, file.text, [extracted_units])

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


    output_path = tasks_path()
    for dataset in ["editorials", "essays", "webDiscourse"]:
        dataset_path = (datasets_path() / "unit-segmentation" / "simple" / dataset)

        dataset_name = DATASET_NAME_TEMPLATE.replace("{dataset}", dataset)
        dataset_file_train = DATASET_FILE_TEMPLATE.replace("{dataset}", dataset).format(split="train")
        dataset_file_test = DATASET_FILE_TEMPLATE.replace("{dataset}", dataset).replace("{split}", "test")
        train_data, test_data = process_folder(dataset_path)
        train_data, test_data = convert_arguments(dataset_name, test_data, train_data)
        if dataset == "essays":
            genre = Genres.ESSAYS
        elif dataset == "editorials":
            genre = Genres.NEWS
        else:
            genre = Genres.WEB_FORUMS
        train_data.append_genre(genre)
        test_data.append_genre(genre)

        train_data.append_subarea(Skills.MINING)

        train_data.write_output(dataset_file_train)
        test_data.write_output(dataset_file_test)
        metadata = Metadata(dataset_name)
        metadata.add_dataset(dataset_file_train, "train")
        metadata.add_dataset(dataset_file_test, "test")
        metadata.add_genre(genre)
        metadata.add_skill(Skills.MINING)
        metadata.write_metadata()
