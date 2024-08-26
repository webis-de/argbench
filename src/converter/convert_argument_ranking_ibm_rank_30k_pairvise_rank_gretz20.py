from common import Output, datasets_path, read_tabular, tasks_path, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import random
import uuid

DATASET_NAME = "argument_ranking_ibm_rank_30k_pairvise_rank_gretz20"

TOLERANCE = 0.05

RANK_MAPPING = [
    "[0] > [1]",
    "[1] > [0]",
    "[0] = [1]"
]

def make_output(dataset, dataset_name):
    output = Output(DATASET_NAME)

    output.append_definition("Rank the following arguments on the given topic according to their quality. All the arguments should be included and listed using identifiers, in descending order of relevance. The output format should be: [0] > [1], [1] > [0] or [0] = [1] if both arguments are equally good. Only respond with the ranking results, do not say any word or explain.")

    for idx, row in dataset.iterrows():
        compare_arg = (dataset[dataset["topic"] == row["topic"]]
                        .drop([idx])
                        .sample(1)
                        .iloc[0])
        prompt = f"Topic: {row['topic']}\nArguments:\n[0] {row['argument']}\n[1] {compare_arg['argument']}"

        id = str(uuid.uuid4())

        if abs(row["WA"] - compare_arg["WA"]) <= TOLERANCE:
            positive_response = RANK_MAPPING[2]
        elif row["WA"] > compare_arg["WA"]:
            positive_response = RANK_MAPPING[0]
        else:
            positive_response = RANK_MAPPING[1]

        negative_response = random.choice([m for m in RANK_MAPPING if m != positive_response])

        output.append_positive_example(prompt, positive_response, "Arguments are ordered based on wighted average quality score")

        output.append_negative_example(prompt, negative_response, "Arguments are ordered based on wighted average quality score but in ascending order")

        output.append_instance(id, prompt, [positive_response])

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

    metadata.write_metadata()
