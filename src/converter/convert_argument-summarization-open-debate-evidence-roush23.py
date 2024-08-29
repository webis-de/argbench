from common import Output, Metadata, add_seed_arg, set_seed, datasets_path
from argparse import ArgumentParser
from datasets import load_dataset
import pyarrow.parquet as pq
import pyarrow as pa
import os

DATASET_NAME = "argument_summarization_open_debate_evidence_roush23"
DATASET_FILE = f"{DATASET_NAME}.json"


def make_output(dataset, metadata, dataset_name):
    output = Output(dataset_name)

    output.append_definition("Given the following argument, generate a short summary.")

    for i, data in enumerate(dataset['train']):
        id = data['id']
        input_text = data['fulltext']
        output_text = data['summary']
        output.append_instance(id, input_text, output_text)

    metadata.add_dataset(dataset_name)
    output.write_output(DATASET_FILE)


def download_dataset(cache_directory):
    parquet_files = [
        f'{cache_directory}/0.parquet',
        f'{cache_directory}/1.parquet',
        f'{cache_directory}/2.parquet',
        f'{cache_directory}/3.parquet',
        f'{cache_directory}/4.parquet',
        f'{cache_directory}/5.parquet',
        f'{cache_directory}/6.parquet',
        f'{cache_directory}/7.parquet',
        f'{cache_directory}/8.parquet'
    ]

    tables = [pq.read_table(file) for file in parquet_files]
    combined_table = pa.concat_tables(tables)

    pq.write_table(combined_table, f'{cache_directory}/combined_train.parquet')
    print("download and combine finished")

if __name__ == "__main__":
    # Argument parser setup
    arg_parser = ArgumentParser(description="Convert OpenCaselist dataset into a JSON format for argument summarization task.")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)
    cache_dir = datasets_path()

    download_dataset(cache_dir)

    dataset = load_dataset('parquet', data_files=os.path.join(cache_dir, "combined_train.parquet"))

    metadata = Metadata(DATASET_NAME)

    make_output(dataset, metadata, DATASET_NAME)

    print("finished make output")
    metadata.add_evaluation_metric("f1_macro")
    metadata.write_metadata()
    print("finished all")

