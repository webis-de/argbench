from common import Output, datasets_path, read_tabular, tasks_path, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import pickle
from dataclasses import dataclass


DATASET_NAME = "stance_classification_argu_saha23"


@dataclass
class Prompt:
    id: str
    prompt: str
    output: str


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Program to convert gretz20 ibm quality dataset into appropriate form")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    metadata = Metadata(DATASET_NAME)

    output_path = tasks_path()

    splits_path = datasets_path() / "argu" / "arg_span_and_scheme_data_keys.pkl"
    data_path = datasets_path() / "argu" / "arg_span_and_scheme_data.pkl"

    arguments = []

    with open(data_path, "rb") as f:
        data = pickle.load(f)
        for id in data:
            id = str(id)
            prompt = f"Argument: {data[id]['text']}"
            output = data[id]["stance"]
            arguments.append(Prompt(id, prompt, output))

    splits_data = open(splits_path, "rb")
    splits = pickle.load(splits_data)

    split_type = "topic-strict"

    model_split_idx = {}
    for split_idx in splits[split_type]:
        for split_model in splits[split_type][split_idx]:
            model_split_idx[split_model] = []
            for id in splits[split_type][split_idx][split_model]:
                model_split_idx[split_model].append(str(id))

    for split_model in model_split_idx:
        output = Output(DATASET_NAME)
        output.append_definition("Judge the sentiment of the argument. Possible responses: pro if argument has positive sentiment or con if argument has negative sentiment.")
        for id in model_split_idx[split_model]:
            for arg in arguments:
                if arg.id == id:
                    output.append_instance(arg.id, arg.prompt, [arg.output])
                    break
        dataset_file = f"stance_classification_argu_{split_model}_saha23.json"
        metadata.add_dataset(dataset_file, split_model)
        output.write_output(dataset_file)

    output = Output(DATASET_NAME)
    output.append_definition("Judge the sentiment of the argument. Possible responses: pro if argument has positive sentiment or con if argument has negative sentiment.")
    for arg in arguments:
        output.append_instance(arg.id, arg.prompt, [arg.output])
    dataset_file = f"stance_classification_argu_full_saha23.json"
    metadata.add_dataset(dataset_file)
    output.write_output(dataset_file)

    splits_data.close()

    metadata.add_evaluation_metric("f1_macro")

    metadata.write_metadata()
