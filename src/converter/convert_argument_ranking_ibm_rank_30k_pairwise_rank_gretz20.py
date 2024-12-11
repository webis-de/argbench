from common import Output, datasets_path, read_tabular, tasks_path, Metadata, add_seed_arg, set_seed, Genres, Subareas
from argparse import ArgumentParser
import random
import uuid

DATASET_NAME = "argument_ranking_ibm_rank_30k_pairvise_rank_gretz20"

TOLERANCE = 0.05

RANK_MAPPING = [
    "Better",
    "Worse",
    "Same"
]

def make_output(dataset, dataset_name):
    output = Output(DATASET_NAME)

    output.append_definition("""Given the following argument pairs, is the first argument better, same, or worse than 
    the second argument in terms of quality. Only respond with better or worse, do not say any word or explain.""")

    for idx, row in dataset.iterrows():
        compare_arg = (dataset[dataset["topic"] == row["topic"]]
                        .drop([idx])
                        .sample(1)
                        .iloc[0])
        prompt = f"Topic: {row['topic']}\nArgument 1: {row['argument']}\nArgument 2: {compare_arg['argument']}"

        id = str(uuid.uuid4())

        if abs(row["WA"] - compare_arg["WA"]) <= TOLERANCE:
            positive_response = RANK_MAPPING[2]
        elif row["WA"] > compare_arg["WA"]:
            positive_response = RANK_MAPPING[0]
        else:
            positive_response = RANK_MAPPING[1]

        output.append_instance(id, prompt, [positive_response])

    output.append_genre(Genres.DEBATES)
    output.append_subarea(Subareas.RANKING)
    output.write_output(dataset_name)

if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Program to convert gretz20 ibm quality dataset into appropriate form")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    metadata = Metadata(DATASET_NAME)

    dataset_path = str(datasets_path()
                    / "ibm-rank-30k"
                    / "arg_quality_rank_30k.csv")

    dataset = read_tabular(dataset_path)
    print("Train")
    make_output(dataset[dataset["set"] == "train"], "argument_ranking_ibm_rank_30k_pairvise_rank_train_gretz20.json")
    print("Test")
    make_output(dataset[dataset["set"] == "test"], "argument_ranking_ibm_rank_30k_pairvise_rank_test_gretz20.json")
    print("Dev")
    make_output(dataset[dataset["set"] == "dev"], "argument_ranking_ibm_rank_30k_pairvise_rank_dev_gretz20.json")

    metadata.add_dataset("argument_ranking_ibm_rank_30k_pairvise_rank_train_gretz20.json", "train")
    metadata.add_dataset("argument_ranking_ibm_rank_30k_pairvise_rank_test_gretz20.json", "test")
    metadata.add_dataset("argument_ranking_ibm_rank_30k_pairvise_rank_dev_gretz20.json", "dev")

    metadata.add_evaluation_metric("f1_macro")
    metadata.add_genre(Genres.DEBATES)
    metadata.add_subarea(Subareas.RANKING)
    metadata.write_metadata()
