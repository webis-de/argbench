from common import Genres, Output, Subareas, datasets_path, Metadata, add_seed_arg, set_seed, read_tabular
from argparse import ArgumentParser


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="What dataset will be processed?")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    data_path = datasets_path() / "political-debates" / "full_dataset.tsv"

    # Set name of the dataset to identify it and files of that dataset
    dataset_name = "argument_relation_identification_political_debates_menini18"
    dataset_file = "argument_relation_identification_political_debates_menini18.json"

    # Class for collecting dataset file data
    # Dataset name specifies folder where dataset will be written
    output = Output(dataset_name)
    output.append_definition(
        """Given the following two arguments on the given topic:\n 
        "Detect whether the first argument supports, attacks, or is unrelated to the second argument.
        Only output support, attack, or unrelated.
        """)

    metadata = Metadata(dataset_name)

    dataset = read_tabular(data_path, "\t")
    dataset["input"] = "Topic: " + dataset["topic"] + "\nArgument 1:" + dataset["argument1"] + "\nArgument 2:" + dataset["argument2"]
    dataset["output"] = dataset["relation"].map({
        "no_relation": "Unrelated",
        "support": "Support",
        "attack": "Attack"
    })

    for row in dataset.iterrows():
        instance = row[1]
        id = instance["pair_id"]
        output.append_instance(id, instance["input"], [instance["output"]])

    metadata.add_dataset(dataset_file)
    metadata.add_evaluation_metric("f1_macro")
    metadata.add_genre(Genres.DEBATES)
    metadata.add_subarea(Subareas.MINING)
    metadata.write_metadata()

    output.append_genre(Genres.DEBATES)
    output.append_subarea(Subareas.MINING)
    output.write_output(dataset_file)
