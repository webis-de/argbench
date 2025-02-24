from common import Genres, Output, Subareas, datasets_path, read_tabular, tasks_path, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import uuid

DATASET_NAME = "argument_ranking_ibm_rank_30k_class_rank_gretz20"

def make_output(dataset, dataset_name):
    output = Output(DATASET_NAME)

    output.append_definition("Judge quality of argument. Possible responses: High Quality if argument is of good quality or Low Quality if argument is of bad quality.")

    for row in dataset.iterrows():
        row = row[1]
        prompt = f"Argument: {row['argument']}"
        response = "High Quality" if row["stance_WA"] == 1 else "Low Quality"
        id = str(uuid.uuid4())

        output.append_instance(id, prompt, [response])

    output.append_genre(Genres.DEBATES)
    output.append_subarea(Subareas.QUALITY_ASSESSMENT)
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
    make_output(dataset[dataset["set"] == "train"], "argument_ranking_ibm_rank_30k_class_rank_train_gretz20.json")
    make_output(dataset[dataset["set"] == "test"], "argument_ranking_ibm_rank_30k_class_rank_test_gretz20.json")
    make_output(dataset[dataset["set"] == "dev"], "argument_ranking_ibm_rank_30k_class_rank_dev_gretz20.json")

    metadata.add_dataset("argument_ranking_ibm_rank_30k_class_rank_train_gretz20.json", "train")
    metadata.add_dataset("argument_ranking_ibm_rank_30k_class_rank_test_gretz20.json", "test")
    metadata.add_dataset("argument_ranking_ibm_rank_30k_class_rank_dev_gretz20.json", "dev")

    
    metadata.add_genre(Genres.DEBATES)
    metadata.add_skill(Subareas.QUALITY_ASSESSMENT)
    metadata.write_metadata()
