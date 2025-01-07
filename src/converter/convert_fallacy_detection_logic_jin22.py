from common import Genres, Output, Subareas, datasets_path, read_tabular, tasks_path, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import uuid


DATASET_NAME = "fallacy_detection_logic_jin22"

def process_file(dataset_path, dataset_file, metadata):
    output = Output(DATASET_NAME)
    output.append_definition("Given article, output which logical fallacies does it have. Possible logical fallacies: " +
                             "faulty generalization - an informal fallacy wherein a conclusion is drawn about all or many instances of a phenomenon on the basis of one or a few instances of that phenomenon. is an example of jumping to conclusions; " +
                             "false causality - statement that jumps to a conclusion implying a causal relationship without supporting evidence; " +
                             "circular reasoning - when the end of an argument comes back to the beginning without having proven itself; " +
                             "ad populum - a fallacious argument which is based on affirming that something is real or better because the majority thinks so; " +
                             "ad hominem -instead of addressing someone's argument or position, you irrelevantly attack the person or some aspect of the person who is making the argument; " +
                             "fallacy of logic - an error in the logical structure of an argument; " +
                             "appeal to emotion - manipulation of the recipient's emotions in order to win an argument; " +
                             "false dilemma - presenting only two options or sides when there are many options or sides; " +
                             "equivocation - when a key term or phrase in an argument is used in an ambiguous way, with one meaning in one portion of the argument and then another meaning in another portion of the argument; " +
                             "fallacy of extension - attacking an exaggerated or caricatured version of your opponent's position; " +
                             "fallacy of relevance - introducing premises or conclusions that have nothing to do with the subject matter; " +
                             "fallacy of credibility - attempts to disprove an argument by attacking the character of the speaker; " +
                             "miscellaneous - miscellaneous; " +
                             "intentional - some intentional (sometimes subconscious) action/choice to incorrectly support an argument.")

    data = read_tabular(dataset_path)

    for row in data.iterrows():
        row = row[1]
        id = row["original_url"]
        prompt = f"{row['source_article']}"
        if "updated_label" in data.columns:
            label = row["updated_label"]
        else:
            label = row["logical_fallacies"]
        output.append_instance(id, prompt, [label])

    output.append_genre(Genres.ESSAYS)
    output.append_subarea(Subareas.REASONING)
    metadata.add_dataset(dataset_file)
    output.write_output(dataset_file)


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="What dataset will be processed?")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args) # Seed random number generation

    data_folder = datasets_path() / "logic"  # path to data
    climate_dataset = data_folder / "climate_all.csv"
    edu_dataset = data_folder / "edu_all.csv"

    # Set name of the dataset to identify it and files of that dataset
    climate_file = "fallacy_detection_logic_climate_jin22.json"
    edu_file = "fallacy_detection_logic_edu_jin22.json"
    metadata = Metadata(DATASET_NAME)

    process_file(climate_dataset, climate_file, metadata)
    process_file(edu_dataset, edu_file, metadata)

    metadata.add_evaluation_metric("f1_macro")

    metadata.add_genre(Genres.ESSAYS)
    metadata.add_subarea(Subareas.REASONING)
    metadata.write_metadata()
