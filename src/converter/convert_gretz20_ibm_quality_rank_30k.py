from common import Output, datasets_path, read_tabular, tasks_path, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import uuid

DATASET_NAME = "gretz20_ibm_quality_rank_30k"

def make_output(dataset, dataset_name):
    output = Output(DATASET_NAME)

    output.append_definition("Judge quality of argument. Possible responses: High Quality if argument is of good quality or Low Quality if argument is of bad quality.")

    for row in dataset.iterrows():
        row = row[1]
        prompt = f"Argument: {row['argument']}"
        response = "High Quality" if row["stance_WA"] == 1 else "Low Quality"
        wrong_response = "Low Quality" if row["stance_WA"] == 1 else "High Quality"
        id = str(uuid.uuid4())
        output.append_positive_example(prompt, response, "")

        output.append_negative_example(prompt, wrong_response, "")

        output.append_instance(id, prompt, [response])

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
    make_output(dataset[dataset["set"] == "train"], "gretz20_ibm_quality_rank_30k_train.json")
    make_output(dataset[dataset["set"] == "test"], "gretz20_ibm_quality_rank_30k_test.json")
    make_output(dataset[dataset["set"] == "dev"], "gretz20_ibm_quality_rank_30k_dev.json")

    metadata.add_dataset("gretz20_ibm_quality_rank_30k_train.json", "train")
    metadata.add_dataset("gretz20_ibm_quality_rank_30k_test.json", "test")
    metadata.add_dataset("gretz20_ibm_quality_rank_30k_dev.json", "dev")

    metadata.add_evaluation_metric("f1_macro")

    metadata.write_metadata()
