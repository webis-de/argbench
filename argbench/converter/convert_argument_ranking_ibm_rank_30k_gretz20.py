from common import Output, datasets_path, read_tabular, tasks_path, Metadata, add_seed_arg, set_seed, Genres, Skills
from argparse import ArgumentParser
import random
import uuid

DATASET_NAME = "argument_ranking_ibm_rank_30k_gretz20"



RANK_MAPPING = [
    "Better",
    "Worse"
]

def make_output(dataset, dataset_name):
    output = Output(DATASET_NAME)

    output.append_definition("""Given the following argument pairs, is the first argument better or worse than 
    the second argument in terms of quality. Only respond with Better or Worse.""")

    for idx, row in dataset.iterrows():
        compare_arg = (dataset[dataset["topic"] == row["topic"]]
                        .drop([idx])
                        .sample(1)
                        .iloc[0])
        prompt = f"Topic: {row['topic']}\nArgument 1: {row['argument']}\nArgument 2: {compare_arg['argument']}"

        id = str(uuid.uuid4())

        if row["WA"] > compare_arg["WA"]:
            positive_response = RANK_MAPPING[0]
        else:
            positive_response = RANK_MAPPING[1]

        output.append_instance(id, prompt, [positive_response])

    output.append_genre(Genres.WIKIPEDIA)
    output.append_subarea(Skills.QUALITY_ASSESSMENT)
    output.write_output(dataset_name)

if __name__ == "__main__":
    arg_parser = ArgumentParser()
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    

    metadata = Metadata(DATASET_NAME)

    dataset_path = str(datasets_path()
                    / "ibm-rank-30k"
                    / "arg_quality_rank_30k.csv")

    dataset = read_tabular(dataset_path)
    print("Train")


    make_output(dataset[dataset["set"] == "train" ], "argument_ranking_ibm_rank_30k_train_gretz20.json")

    make_output(dataset[dataset["set"] == "dev" ], "argument_ranking_ibm_rank_30k_val_gretz20.json")

    make_output(dataset[dataset["set"] == "test"], "argument_ranking_ibm_rank_30k_test_gretz20.json")


    metadata.add_dataset("argument_ranking_ibm_rank_30k_train_gretz20.json", "train")
    metadata.add_dataset("argument_ranking_ibm_rank_30k_test_gretz20.json", "test")
    metadata.add_dataset("argument_ranking_ibm_rank_30k_val_gretz20.json", "val")


    metadata.add_evaluation_metric("fscore")
    metadata.add_genre(Genres.WIKIPEDIA)
    metadata.add_skill(Skills.QUALITY_ASSESSMENT)
    metadata.write_metadata()
