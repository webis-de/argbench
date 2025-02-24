from common import Output, Metadata, add_seed_arg, set_seed, datasets_path
from argparse import ArgumentParser
from datasets import load_dataset

DATASET_NAME = "argument_summarization_open_debate_evidence_roush23"


def make_output(dataset, metadata, dataset_name, output_file):
    output = Output(DATASET_NAME)

    output.append_definition("Given the following argument, generate a short summary.")

    for i, data in enumerate(dataset['train']):
        id = data['id']
        input_text = data['fulltext']
        output_text = data['tag']
        if not input_text or not output_text:
            continue
        output.append_instance(id, input_text, output_text)
    metadata.add_dataset(dataset_name)
    output.write_output(output_file)


def download_and_process_datasets(cache_directory):
    for number in range(8):
        parquet_file = f'{cache_directory}/{number}.parquet'
        dataset = load_dataset('parquet', data_files=parquet_file)
        output_file = f"{DATASET_NAME}_with_tag_cleaned_{number}.json"
        metadata = Metadata(output_file)
        make_output(dataset, metadata, DATASET_NAME, output_file)
        
        metadata.write_metadata()
        print(f"Finished processing {parquet_file}")


if __name__ == "__main__":
    # Argument parser setup
    arg_parser = ArgumentParser(description="Convert OpenCaselist dataset into a JSON format for argument summarization task.")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)
    cache_dir = datasets_path()
    download_and_process_datasets(cache_dir)

    print("Finished all tasks")

