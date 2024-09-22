from common import Genres, Output, Subareas, datasets_path, read_tabular, tasks_path, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import random
import uuid

DATASET_NAME = "argument_similarity_ukp_aspect_reimers19"

SIMILARITY_MAPPING = {
    "HS": "High Similarity",
    "SS": "Some Similarity",
    "NS": "No Similarity",
    "DTORCD": "Different Topic/Can't Decide"
}

if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Program to convert gretz20 ibm quality dataset into appropriate form")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    metadata = Metadata(DATASET_NAME)

    dataset_path = str(datasets_path()
                       / "ukp-aspect-argument-similarity"
                       / "UKP_ASPECT.tsv")

    dataset = read_tabular(dataset_path, separator="\t")
    dataset = dataset.dropna()

    output = Output(DATASET_NAME)

    output.append_definition("Judge if the argument pairs are similar (on the same topic and cover similar aspects) or dissimilar (e.g., cover different topics or different aspects). Possible outputs: High similarity if two arguments are very similar, Some Similarity if two arguments are somewhat similar, No Similarity if two arguments are not similar, Different Topic/Can't Decide if two arguments belong to different topics.")

    for row in dataset.iterrows():
        row = row[1]
        prompt = f"Argument 1: {row['sentence_1']}\nArgument 2: {row['sentence_2']}"
        response = SIMILARITY_MAPPING[row["label"]]
        id = str(uuid.uuid4())

        output.append_instance(id, prompt, [response])

    output.append_genre(Genres.ESSAYS)
    output.append_subarea(Subareas.REASONING)
    output.write_output("argument_similarity_ukp_aspect_reimers19.json")
    metadata.add_dataset("argument_similarity_ukp_aspect_reimers19.json")
    metadata.add_genre(Genres.ESSAYS)
    metadata.add_subarea(Subareas.REASONING)
    metadata.add_evaluation_metric("f1_macro")
    metadata.write_metadata()
