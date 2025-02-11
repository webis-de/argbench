from common import Output, datasets_path, read_tabular, tasks_path, Metadata, add_seed_arg, set_seed, Genres, Subareas
from argparse import ArgumentParser
import uuid
import pandas as pd

DATASET_NAME = "argument_ranking_ibm_rank_30k_gretz20"

def make_output(dataset, dataset_name):
    output = Output(DATASET_NAME)

    output.append_definition("Rank the following arguments on the given topic according to their quality. All the arguments should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., [4] > [2]. Only respond with the ranking results, do not say any word or explain.")

    for idx, row in dataset.iterrows():
        compare_args = (dataset[dataset["topic"] == row["topic"]]
                        .drop([idx])
                        .sample(10))
        prompt = f"Topic: {row['topic']}\nArguments:\n[0] {row['argument']}\n"
        for compare_idx, compare_row in enumerate(compare_args.iterrows()):
            compare_row = compare_row[1]
            prompt += f"[{compare_idx + 1}] {compare_row['argument']}\n"

        id = str(uuid.uuid4())

        total_rows = pd.concat([pd.DataFrame([row]), compare_args], ignore_index=True)

        positive_response = " > ".join(total_rows
                                       .sort_values(by=["WA"], ascending=False)
                                       .reset_index()["index"]
                                       .map("[{}]".format))


        output.append_instance(id, prompt, [positive_response])

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
    print("Train")
    make_output(dataset[dataset["set"] == "train"], "argument_ranking_ibm_rank_30k_full_rank_train_gretz20.json")
    print("Test")
    make_output(dataset[dataset["set"] == "test"], "argument_ranking_ibm_rank_30k_full_rank_test_gretz20.json")
    print("Dev")
    make_output(dataset[dataset["set"] == "dev"], "argument_ranking_ibm_rank_30k_full_rank_dev_gretz20.json")

    metadata.add_dataset("argument_ranking_ibm_rank_30k_full_rank_train_gretz20.json", "train")
    metadata.add_dataset("argument_ranking_ibm_rank_30k_full_test_gretz20.json", "test")
    metadata.add_dataset("argument_ranking_ibm_rank_30k_full_dev_train_gretz20.json", "dev")

    metadata.add_evaluation_metric("f1_macro")
    metadata.add_genre(Genres.DEBATES)
    metadata.add_subarea(Subareas.QUALITY_ASSESSMENT)
    metadata.write_metadata()
