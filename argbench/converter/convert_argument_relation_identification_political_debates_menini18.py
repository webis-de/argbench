from common import Genres, Output, Skills, datasets_path, Metadata, add_seed_arg, set_seed, read_tabular, \
    find_topic_size_to_split
from argparse import ArgumentParser
from random import sample


def process_split(dataset, path):
    output = Output(dataset_name)
    output.append_definition(
    """Given the following two arguments on the given topic:\n 
    "Detect whether the first argument supports, attacks, or is unrelated to the second argument.
    Only output support, attack, or unrelated.
    """)
    for row in dataset.iterrows():
        instance = row[1]
        id = instance["pair_id"]
        output.append_instance(id, instance["input"], [instance["output"]])
    output.append_genre(Genres.DEBATES)
    output.append_subarea(Skills.MINING)
    output.write_output(path)

if __name__ == "__main__":
    arg_parser = ArgumentParser()
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    data_path = datasets_path() / "political-debates" / "balanced_dataset.tsv"

    # Set name of the dataset to identify it and files of that dataset
    dataset_name = "argument_relation_identification_political_debates_menini18"
    dataset_file = "argument_relation_identification_political_debates_{split}_menini18.json"

    # Class for collecting dataset file data
    # Dataset name specifies folder where dataset will be written

    metadata = Metadata(dataset_name)
    metadata.add_dataset(dataset_file.format(split="train"), "train")
    metadata.add_dataset(dataset_file.format(split="test"), "test")
    metadata.add_dataset(dataset_file.format(split="val"), "val")

    metadata.add_evaluation_metric("fscore")
    metadata.add_genre(Genres.DEBATES)
    metadata.add_skill(Skills.MINING)
    metadata.write_metadata()


    dataset = read_tabular(data_path, "\t")
    dataset["input"] = "Topic: " + dataset["topic"] + "\nArgument 1:" + dataset["argument1"] + "\nArgument 2:" + dataset["argument2"]
    dataset["output"] = dataset["relation"].map({
        "no_relation": "Unrelated",
        "support": "Support",
        "attack": "Attack"
    })

    df_test, df_train = find_topic_size_to_split(dataset, "topic", 0.2)
    df_val, df_train = find_topic_size_to_split(df_train, "topic", 0.25)
    print(len(df_train))
    print(len(df_test))
    print(len(dataset))

    process_split(df_train, dataset_file.format(split="train"))
    process_split(df_test, dataset_file.format(split="test"))
    process_split(df_val, dataset_file.format(split="val"))


