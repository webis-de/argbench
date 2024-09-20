from common import Genres, Output, Subareas, datasets_path, read_tabular, tasks_path, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import ast
import uuid


if __name__ == "__main__":
    # Input arguments for dataset generation
    arg_parser = ArgumentParser(description="What dataset will be processed?")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args) # Seed random number generation

    data_path = datasets_path() / "kialo" / "kialo-pta24.csv" # path to data

    # Set name of the dataset to identify it and files of that dataset
    dataset_name = "argument_canonicalization_kialo_saad-yazdi24"
    dataset_file = "argument_canonicalization_kialo_saad-yazdi24.json"

    # Class for collecting dataset file data
    # Dataset name specifies folder where dataset will be written
    output = Output(dataset_name)
    output.append_definition("Rewrite argument in appropriate canonical form. Possible canonical forms: " +
                             "Alpha: a is X, because a is Y; Beta: a is X, because b is X; Gamma: a is X, because b is Y; Delta: q is true, because q is R;")

    dataset = read_tabular(data_path)

    metadata = Metadata(dataset_name)
    #####
    # Read dataset
    for row in dataset.iterrows():
        row = row[1]

        id = row["pairid"]

        canon_form = ast.literal_eval(row["canonical_form"])

        prompt = f"Topic: {row['topic']}\nPremise: {row['premise']}\nConclusion: {row['conclusion']}"

        if row["form"] == "alpha":
            canon_str = f"{canon_form['a']} is {canon_form['x']}, because {canon_form['a']} is {canon_form['y']}"
        elif row["form"] == "beta":
            canon_str = f"{canon_form['a']} is {canon_form['x']}, because {canon_form['b']} is {canon_form['x']}"
        elif row["form"] == "gamma":
            canon_str = f"{canon_form['a']} is {canon_form['x']}, because {canon_form['b']} is {canon_form['y']}"
        elif row["form"] == "delta":
            canon_str = f"{canon_form['a']} is true, because {canon_form['a']} is {canon_form['r']}"

        output.append_instance(id, prompt, [canon_str])

    output.append_genre(Genres.DEBATE_PORTALS)
    output.append_subarea(Subareas.REASONING)
    output.write_output(dataset_file)

    metadata.add_evaluation_metric("rouge")

    metadata.add_dataset(dataset_file)
    metadata.add_genre(Genres.DEBATE_PORTALS)
    metadata.add_subarea(Subareas.REASONING)
    metadata.write_metadata()
