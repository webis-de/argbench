from common import Output, datasets_path, tasks_path, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import re
import ndjson

dataset_name = "aspect_detection_ukp_aspect_schiller21"

def process_split(dataset_file, output_file, metadata, dataset_split):
    output = Output(dataset_name)
    output.append_definition("Extract aspects of the argument. Aspect is a small substring of original text that can characterize the argument.")

    with open(dataset_file, "r") as f:
        dataset = ndjson.load(f)

        for row in dataset:
            id = row["hash"]
            argument = row["sentence"]
            aspect_pos_string = row["aspect_pos_string"]

            prompt = f"Argument: {argument}"
            aspect_output = "\n".join([asp_text for asp_text in aspect_pos_string])

            output.append_instance(id, prompt, [aspect_output])

    output.write_output(output_file)
    metadata.add_dataset(output_file, dataset_split=dataset_split)


if __name__ == "__main__":
    # Input arguments for dataset generation
    arg_parser = ArgumentParser(description="What dataset will be processed?")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args) # Seed random number generation

    data_path = datasets_path() / "ukp-aspect-argument-generation" / "argument_aspect_detection_v1.0" / "in_topic"

    metadata = Metadata(dataset_name)

    process_split(
        data_path / "train.jsonl",
        "aspect_detection_ukp_aspect_train_schiller21.json",
        metadata,
        "train"
    )
    process_split(
        data_path / "test.jsonl",
        "aspect_detection_ukp_aspect_test_schiller21.json",
        metadata,
        "test"
    )
    process_split(
        data_path / "dev.jsonl",
        "aspect_detection_ukp_aspect_dev_schiller21.json",
        metadata,
        "dev"
    )

    metadata.add_evaluation_metric("rouge")
    metadata.write_metadata()
