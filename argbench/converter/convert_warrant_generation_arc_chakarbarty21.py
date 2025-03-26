#!/usr/bin/env python3
from common import Genres, Output, Skills, datasets_path, tasks_path, Metadata, add_seed_arg, set_seed, split_test_train
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
    output_para_comet = Output(para_comet_name)

    target_output.append_definition("Given a premise and a claim, generate an enthymem. An enthymem is a reason with which the claim follows logically form the premise.  Do not explain.")
    output_para_comet.append_definition("Given a premise and a claim, generate an enthymem using provided context.  An enthymem is a reason with which the claim follows logically form the premise. Do not explain.")

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
        metadata.add_skill(subarea)
    metadata.add_evaluation_metric("generation-score")

    return target_output, output_para_comet


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    root_data_path = Path(datasets_path() / "warrant-generation-chakrabarty21" )

    output_path = tasks_path()

    datsaet_target = {
        "arc": "D1test/semevaldata.target",
        "ideological_debate": "D2test/jandataparacomet.hypo",
        "microtext_1": "D3test/ikatdata.target",
        "art_train" :  "enthymemes-paracomet1/train.target",
        "art_test"   :  "enthymemes-paracomet1/val.target"
    }

    dataset_source = {
        "arc": "D1test/semevaldataparacomet.source",
        "ideological_debate": "D2test/jandataparacomet.source",
        "microtext_1": "D3test/ikatdataparacomet.source",
        "art_train" :  "enthymemes-paracomet1/train.source",
        "art_test"   :  "enthymemes-paracomet1/val.source"

    }

    for dataset in ["arc", "ideological_debate", "microtext_1", "art"]:
        dataset_name = f"warrant_generation_{dataset}_chakarbarty21"
        dataset_para_name = f"warrant_generation_{dataset}_para_comet_chakarbarty21"
        if dataset == "arc":
            genre = Genres.NEWS
        elif dataset == "arg":
            genre = Genres.STORIES
        elif dataset == "ideological_debate":
            genre = Genres.DEBATE_PORTALS
        else:
            genre = Genres.ESSAYS
        metadata = Metadata(dataset_name)
        if dataset == "art":
            train_data_path = root_data_path /datsaet_target["art_train"]
            test_data_path =  root_data_path /datsaet_target["art_test"]
            train_data_para_path =  root_data_path /dataset_source["art_train"]
            test_data_para_path = root_data_path /dataset_source["art_test"]
            train_data, train_para_data = read_para_comet(train_data_path, train_data_para_path)
            test_data, test_para_data = read_para_comet(test_data_path, test_data_para_path)
            data_set = {"test": test_data, "train": train_data}
            data_para_set = {"test": test_para_data, "train": train_para_data}
        else:
            data_path = root_data_path / datsaet_target[dataset]
            para_comet_path = root_data_path / dataset_source[dataset]
            data, para_comet_data = read_para_comet(data_path, para_comet_path)
            test_data, train_data = split_test_train(data)
            test_para_data, train_para_data = split_test_train(para_comet_data)
            data_set = {"test": test_data, "train": train_data}
            data_para_set = {"test": test_para_data, "train": train_para_data}
        for split in ["test", "train"]:
            dataset_file_name = f"warrant_generation_{dataset}_{split}_chakarbarty21.json"
            dataset_para_file_name = f"warrant_generation_{dataset}_para_comet_{split}_chakarbarty21.json"

            metadata.add_dataset(dataset_file_name, split)
            data, para_comet_data = make_outputs(
                data_set[split],
                data_para_set[split],
                dataset_name,
                dataset_para_name,
                metadata,
                [genre],
                [Skills.GENERATION]
            )

            data.write_output(dataset_file_name)
            para_comet_data.write_output(dataset_para_file_name)
        metadata.write_metadata()


