#!/usr/bin/env python3
from common import Output, datasets_path, tasks_path, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
from dataclasses import dataclass
import uuid
import re

DATASET_NAME = "premise_generation_arc_chakarbarty21"

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


def make_outputs(target_data, para_comet_data):
    target_output = Output(DATASET_NAME)
    output_para_comet = Output(DATASET_NAME)

    target_output.append_definition("Given a reason and a claim, generate an enthymeme.")
    output_para_comet.append_definition("Given a reason and a claim, generate an enthymeme using ptovided context.")

    for i in range(len(target_data)):
        prompt = f"Reason: {target_data[i].reason}\nClaim: {target_data[i].claim}"
        prompt_para_comet = f"Reason: {target_data[i].reason}\nContext: {para_comet_data[i].enthymeme}\nClaim: {target_data[i].claim}"
        output = target_data[i].enthymeme
        id = str(uuid.uuid4())
        target_output.append_instance(id, prompt, [output])
        output_para_comet.append_instance(id, prompt_para_comet, [output])

    return target_output, output_para_comet


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Program to convert ajjour unit segmentation dataset into appropriate form")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    data_path = (datasets_path()/"art")

    output_path = tasks_path()

    metadata = Metadata(DATASET_NAME)

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

    d1test_data, d1test_data_para_comet = read_para_comet(d1test_data, d1test_data_para_comet)
    d1test_data, d1test_data_para_comet = make_outputs(d1test_data, d1test_data_para_comet)

    d1test_data.write_output("premise_generation_arc_d1test_chakarbarty21.json")
    d1test_data_para_comet.write_output("premise_generation_arc_d1test_para_comet_chakarbarty21.json")

    metadata.add_dataset("premise_generation_arc_d1test_chakarbarty21.json", "test")
    metadata.add_dataset("premise_generation_arc_d1test_para_comet_chakarbarty21.json", "test")

    d2test_data, d2test_data_para_comet = read_para_comet(d2test_data, d2test_data_para_comet)
    d2test_data, d2test_data_para_comet = make_outputs(d2test_data, d2test_data_para_comet)

    d2test_data.write_output("premise_generation_arc_d2test_chakarbarty21.json")
    d2test_data_para_comet.write_output("premise_generation_arc_d2test_para_comet_chakarbarty21.json")

    metadata.add_dataset("premise_generation_arc_d2test_chakarbarty21.json", "test")
    metadata.add_dataset("premise_generation_arc_d2test_para_comet_chakarbarty21.json", "test")

    d3test_data, d3test_data_para_comet = read_para_comet(d3test_data, d3test_data_para_comet)
    d3test_data, d3test_data_para_comet = make_outputs(d3test_data, d3test_data_para_comet)

    d3test_data.write_output("premise_generation_arc_d3test_chakarbarty21.json")
    d3test_data_para_comet.write_output("premise_generation_arc_d3test_para_comet_chakarbarty21.json")

    metadata.add_dataset("premise_generation_arc_d3test_chakarbarty21.json", "test")
    metadata.add_dataset("premise_generation_arc_d3test_para_comet_chakarbarty21.json", "test")

    art_data_train_target, art_data_train_target_para_comet = read_para_comet(art_data_train_target, art_data_train_source)
    art_data_train_target, art_data_train_target_para_comet = make_outputs(art_data_train_target, art_data_train_target_para_comet)

    art_data_train_target.write_output("premise_generation_arc_train_chakarbarty21.json")
    art_data_train_target_para_comet.write_output("premise_generation_arc_train_para_comet_chakarbarty21.json")

    metadata.add_dataset("premise_generation_arc_train_chakarbarty21.json", "train")
    metadata.add_dataset("premise_generation_arc_train_para_comet_chakarbarty21.json", "train")

    art_data_val_target, art_data_val_target_para_comet = read_para_comet(art_data_val_target, art_data_val_source)
    art_data_val_target, art_data_val_target_para_comet = make_outputs(art_data_val_target, art_data_val_target_para_comet)

    art_data_val_target.write_output("premise_generation_arc_val_chakarbarty21.json")
    art_data_val_target_para_comet.write_output("premise_generation_arc_val_para_comet_chakarbarty21.json")

    metadata.add_dataset("premise_generation_arc_val_chakarbarty21.json", "val")
    metadata.add_dataset("premise_generation_arc_val_para_comet_chakarbarty21.json", "val")

    metadata.add_evaluation_metric("rouge")

    metadata.write_metadata()
