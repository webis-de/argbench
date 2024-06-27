from common import Output, datasets_path, tasks_path, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import uuid


if __name__ == "__main__":
    # Input arguments for dataset generation
    arg_parser = ArgumentParser(description="What dataset will be processed?")
    add_seed_arg(arg_parser)
    #####
    # Add additional arguments if needed
    arg_parser.add_argument("-a", "--custom_argument", help="Your custom argument")
    #####
    args = arg_parser.parse_known_args()[0]
    set_seed(args) # Seed random number generation

    data_path = datasets_path() / "path/in/dataset/folder" # path to data

    # Set name of the dataset to identify it and files of that dataset
    dataset_name = "dataset_name"
    dataset_file = "dataset_name.json"

    # Class for collecting dataset file data
    # Dataset name specifies folder where dataset will be written
    output = Output(dataset_name)
    output.append_definition("Dataset definition prompt here")

    metadata = Metadata(dataset_name)
    #####
    # Read dataset
    for line in dataset:
        # Add positive and negative examples
        output.append_positive_example(line["input"], line["output"], "Explanation")
        output.append_negative_example(line["input"], line["output"], "Explanation")


        # Add dataset instance, id can be taken from dataset
        id = str(uuid.uuid4())
        output.append_instance(id, line["input"], [line["output"]])
    #####

    # Write processed dataset onto disc
    # One dataset can have multiple Outputs
    # Outputs with the same dataset name will be written in same folder
    output.write_output(dataset_file)

    # Populate dataset metadata
    # Available dataset metrics:
    # - f1_macro
    # - f1_micro
    # - rouge
    # - kendalltau
    metadata.add_evaluation_metric("eval_metric")
    # add_dataset must be called for each Output
    # datasets original split information can also be provided
    # Available splits:
    # - train
    # - test
    # - dev
    # - none (leave split empty)
    # metadata.add_dataset(dataset_file, dataset_split="train")
    metadata.add_dataset(dataset_file)
    metadata.write_metadata()
