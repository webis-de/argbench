import os
import pandas as pd
from common import Output, Metadata, add_seed_arg, set_seed, datasets_path
from argparse import ArgumentParser
from datasets import load_dataset
from tqdm import tqdm
DATASET_NAME = "argument_summarization_open_debate_evidence_roush23"


def make_output(dataset, metadata, dataset_name, output_file):
    output = Output(DATASET_NAME)

    output.append_definition("Given the following argument, generate a short summary.")

    for i, data in dataset.iterrows():
        id = data['id']
        input_text = data['fulltext']
        output_text = data['tag']
        if not input_text or not output_text:
            continue
        output.append_instance(id, input_text, output_text)
    metadata.add_dataset(dataset_name)
    output.write_output(output_file)


def process_dataset(cache_directory):
    for file in tqdm(os.listdir(cache_directory)):
        if file.endswith("csv"):
            df = pd.read_csv(os.path.join(cache_directory,file))
            df = df.sample(100000)
            output_file = f"{DATASET_NAME}_with_tag_cleaned_{file}.json"
            metadata = Metadata(output_file)
            metadata.add_evaluation_metric("f1_macro")
            metadata.write_metadata()
            make_output(df, metadata, DATASET_NAME, output_file)

if __name__ == "__main__":
    # Argument parser setup
    arg_parser = ArgumentParser(description="Convert OpenCaselist dataset into a JSON format for argument summarization task.")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)
    cache_dir = "/bigwork/nhwpajjy/computational-argumentation-tasks-instructions/datasets/openDebateEvidence/datasets--Yusuf5--OpenCaselist/snapshots/751ef23038d6beca927a66c4af5fb8122f2806b5"
    process_dataset(cache_dir)

    print("Finished all tasks")

