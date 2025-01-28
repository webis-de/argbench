#!/usr/bin/env python3
from common import Genres, Output, Subareas, datasets_path, tasks_path, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import uuid
import re

DATASET_ARC_NAME = "warrant_generation_arc_chakarbarty21" # D1
DATASET_ART_NAME = "warrant_generation_art_chakarbarty21" # training
DATASET_MICROTEXT_NAME = "warrant_generation_microtext_1_chakarbarty21" # D3
DATASET_IDEOLOGICAL_DEBATE_NAME = "warrant_generation_ideological_debate_chakarbarty21" # D2


@dataclass
class Record:
    reason: str
    enthymeme: str
    claim: str


def read_para_comet(path, para_comet_path):
    splitter = re.compile(r" #+ ")
    comet_records = []
    target_records = []
    data_file = open(path, "r")
    data_file_comet = open(para_comet_path, "r")
    for line in data_file_comet:
        comet_string = splitter.split(line.strip())
        target_string = next(data_file)
        target_start = len(comet_string[0]) + 1
        try:
            target_end = target_string.lower().index(comet_string[2].lower()) - 2
        except:
            print("===== Error =====") # Skip paraphrase errors
            print(path)

            metadata.write_metadata
            print(comet_string)
            print(target_string)
        comet_record = Record(comet_string[0], comet_string[1], comet_string[2])
        target_record = Record(comet_string[0], target_string[target_start:target_end], comet_string[2])
        comet_records.append(comet_record)
        target_records.append(target_record)
    data_file.close()
    data_file_comet.close()
    return target_records, comet_records


def make_outputs(target_data, para_comet_data, dataset_name, para_comet_name, metadata, genres = None, subareas = None):
    if not genres:
        genres = []
    if not subareas:
        subareas = []

    target_output = Output(dataset_name)
    output_para_comet = Output(dataset_name)

    target_output.append_definition("Given a premise and a claim, generate an enthymem. An enthymem is a reason with which the claim follows logically form the premise.")
    output_para_comet.append_definition("Given a premise and a claim, generate an enthymem using provided context.  An enthymem is a reason with which the claim follows logically form the premise.")

    for i in range(len(target_data)):
        prompt = f"Premise: {target_data[i].reason}\nClaim: {target_data[i].claim}"
        prompt_para_comet = f"Premise: {target_data[i].reason}\nContext: {para_comet_data[i].enthymeme}\nClaim: {target_data[i].claim}"
        output = target_data[i].enthymeme
        id = str(uuid.uuid4())
        target_output.append_instance(id, prompt, [output])
        output_para_comet.append_instance(id, prompt_para_comet, [output])

    for genre in genres:
        target_output.append_genre(genre)
        output_para_comet.append_genre(genre)
        metadata.add_genre(genre)

    for subarea in subareas:
        target_output.append_subarea(subarea)
        output_para_comet.append_subarea(subarea)
        metadata.add_subarea(subarea)

    return target_output, output_para_comet


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Program to convert ajjour unit segmentation dataset into appropriate form")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    data_path = Path("/bigwork/nhwpajjy/computational-argumentation-tasks-instructions/datasets/warrant-generation-chakrabarty21")

    output_path = tasks_path()

    d1test_data = data_path / "D1test" / "semevaldata.target"
    d1test_data_para_comet = data_path / "D1test" / "semevaldataparacomet.source"

    d2test_data = data_path / "D2test" / "jandataparacomet.hypo"
    d2test_data_para_comet = data_path / "D2test" / "jandataparacomet.source"

    d3test_data = data_path / "D3test" / "ikatdata.target"
    d3test_data_para_comet = data_path / "D3test" / "ikatdataparacomet.source"

    art_data_train_source = data_path / "enthymemes-paracomet1" / "train.source"
    art_data_train_target = data_path / "enthymemes-paracomet1" / "train.target"

    art_data_val_source = data_path / "enthymemes-paracomet1" / "val.source"
    art_data_val_target = data_path / "enthymemes-paracomet1" / "val.target"

    # D1test arc

    d1test_name = "warrant_generation_arc_chakarbarty21"
    d1test_para_comet_name = "warrant_generation_arc_para_comet_chakarbarty21"

    d1test_file = d1test_name + ".json"
    d1test_para_comet_file = d1test_para_comet_name + ".json"

    metadata = Metadata(d1test_name)

    d1test_data, d1test_data_para_comet = read_para_comet(d1test_data, d1test_data_para_comet)
    d1test_data, d1test_data_para_comet = make_outputs(
        d1test_data,
        d1test_data_para_comet,
        d1test_name,
        d1test_para_comet_name,
        metadata,
        [Genres.DEBATES],
        [Subareas.MINING]
    )

    d1test_data.write_output(d1test_file)
    d1test_data_para_comet.write_output(d1test_para_comet_file)

    metadata.add_dataset(d1test_file)
    metadata.add_dataset(d1test_para_comet_file)

    metadata.write_metadata()
    # D2test ideological_debate

    d2test_name = "warrant_generation_ideological_debate_chakarbarty21"
    d2test_para_comet_name = "warrant_generation_ideological_debate_para_comet_chakarbarty21"

    d2test_file = d1test_name + ".json"
    d2test_para_comet_file = d1test_para_comet_name + ".json"

    metadata = Metadata(d2test_name)
    d2test_data, d2test_data_para_comet = read_para_comet(d2test_data, d2test_data_para_comet)
    d2test_data, d2test_data_para_comet = make_outputs(
        d2test_data,
        d2test_data_para_comet,
        d2test_name,
        d2test_para_comet_name,
        metadata,
        [Genres.DEBATE_PORTALS],
        [Subareas.MINING]
    )

    d2test_data.write_output(d2test_file)
    d2test_data_para_comet.write_output(d2test_para_comet_file)

    metadata.add_dataset(d2test_file)
    metadata.add_dataset(d2test_para_comet_file)

    metadata.write_metadata()
    # D3 microtext_1

    d3test_name = "warrant_generation_microtexts_1_chakarbarty21"
    d3test_para_comet_name = "warrant_generation_microtexts_1_para_comet_chakarbarty21"

    d3test_file = d3test_name + ".json"
    d3test_para_comet_file = d3test_para_comet_name + ".json"

    metadata = Metadata(d3test_name)
    d3test_data, d3test_data_para_comet = read_para_comet(d3test_data, d3test_data_para_comet)
    d3test_data, d3test_data_para_comet = make_outputs(
        d3test_data,
        d3test_data_para_comet,
        d3test_name,
        d3test_para_comet_name,
        metadata,
        [Genres.DEBATES],
        [Subareas.MINING]
    )

    d3test_data.write_output(d3test_file)
    d3test_data_para_comet.write_output(d3test_para_comet_file)

    metadata.add_dataset(d3test_file)
    metadata.add_dataset(d3test_para_comet_file)

    metadata.write_metadata()
    # training art dataset
    train_name = "warrant_generation_art_chakarbarty21"
    train_para_comet_name = "warrant_generation_art_para_comet_chakarbarty21"

    train_train_file = "warrant_generation_art_train_chakarbarty21.json"
    train_train_para_comet_file = "warrant_generation_art_para_comet_train_chakarbarty21.json"

    train_val_file = "warrant_generation_art_val_chakarbarty21.json"
    train_val_para_comet_file = "warrant_generation_art_para_comet_val_chakarbarty21.json"

    metadata = Metadata(train_name)
    art_data_train_target, art_data_train_target_para_comet = read_para_comet(art_data_train_target, art_data_train_source)
    art_data_train_target, art_data_train_target_para_comet = make_outputs(
        art_data_train_target,
        art_data_train_target_para_comet,
        train_name,
        train_para_comet_name,
        metadata,
        [Genres.DEBATES],
        [Subareas.GENERATION]
    )

    art_data_train_target.write_output(train_train_file)
    art_data_train_target_para_comet.write_output(train_train_para_comet_file)

    metadata.add_dataset(train_train_file, "train")
    metadata.add_dataset(train_train_para_comet_file, "train")

    art_data_val_target, art_data_val_target_para_comet = read_para_comet(art_data_val_target, art_data_val_source)
    art_data_val_target, art_data_val_target_para_comet = make_outputs(
        art_data_val_target,
        art_data_val_target_para_comet,
        train_name,
        train_para_comet_name,
        metadata,
        [Genres.DEBATES],
        [Subareas.GENERATION]
    )

    art_data_val_target.write_output(train_val_file)
    art_data_val_target_para_comet.write_output(train_val_para_comet_file)

    metadata.add_dataset(train_val_file, "val")
    metadata.add_dataset(train_val_para_comet_file, "val")

    metadata.write_metadata()
