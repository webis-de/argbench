from common import Output, datasets_path, read_tabular, tasks_path, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import uuid
import pandas as pd

DATASET_NAME = "gretz20_ibm_quality_rank_30k_rank"

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

        negative_response = " > ".join(total_rows
                                       .sort_values(by=["WA"], ascending=True)
                                       .reset_index()["index"]
                                       .map("[{}]".format))

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
                    / "argument-quality"
                    / "gretz20-a-large-scale-dataset-for-argument-quality-ranking-construction-and-analysis"
                    / "arg_quality_rank_30k.csv")

    dataset = read_tabular(dataset_path)
    print("Train")
    make_output(dataset[dataset["set"] == "train"], "gretz20_ibm_quality_rank_30k_rank_train.json")
    print("Test")
    make_output(dataset[dataset["set"] == "test"], "gretz20_ibm_quality_rank_30k_rank_test.json")
    print("Dev")
    make_output(dataset[dataset["set"] == "dev"], "gretz20_ibm_quality_rank_30k_rank_dev.json")

    metadata.add_dataset("gretz20_ibm_quality_rank_30k_rank_train.json", "train")
    metadata.add_dataset("gretz20_ibm_quality_rank_30k_rank_test.json", "test")
    metadata.add_dataset("gretz20_ibm_quality_rank_30k_rank_dev.json", "dev")

    metadata.add_evaluation_metric("f1_macro")

    metadata.write_metadata()
