from common import Output, datasets_path, read_tabular, tasks_path, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import ast
import uuid

def process_dataset(bows_dataset, metadata, dataset_name, dataset_file, dataset_path):
    dataset = read_tabular(dataset_path)
    dataset["parsed_issues"] = dataset["big_issues"].str.findall("\d")
    output = Output(dataset_name)
    output.append_definition("Given discussion topic and a collection of keywords that describe your stance on various issues, generate a claim that is based on your stance.")

    for row in dataset.iterrows():
        row = row[1]
        total_words = []
        for i, issue_label in enumerate(row["parsed_issues"]):
            if issue_label == "1":
                total_words += bows_dataset["con_parsed"][i]
            if issue_label == "3":
                total_words += bows_dataset["pro_parsed"][i]

        total_words = ", ".join(total_words)
        prompt = f"Topic: {row['topic']}\nStance keywords: {total_words}"
        label = row["top_claim"]
        id = str(uuid.uuid4())
        output.append_instance(id, prompt, [label])

    output.write_output(dataset_file)
    metadata.add_dataset(dataset_file)


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="What dataset will be processed?")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args) # Seed random number generation

    bows_path = (datasets_path() /
                 "belief-arguments" /
                 "data" /
                 "big_issues_bows")

    data_path = (datasets_path() /
                 "belief-arguments" /
                 "data" /
                 "preprocessed_data")

    dataset_name = "conclusion_generation_belief_generation_alshomary21"

    bows_dataset = read_tabular(bows_path / "big_issues_new_bows.csv")
    bows_dataset["pro_parsed"] = bows_dataset["pro_bow"].apply(ast.literal_eval)
    bows_dataset["con_parsed"] = bows_dataset["con_bow"].apply(ast.literal_eval)

    metadata = Metadata(dataset_name)

    process_dataset(
        bows_dataset,
        metadata,
        dataset_name,
        "conclusion_generation_belief_generation_train_alshomary21.json",
        data_path / "train_with_claim_df.csv"
    )

    process_dataset(
        bows_dataset,
        metadata,
        dataset_name,
        "conclusion_generation_belief_generation_test_alshomary21.json",
        data_path / "test_with_claim_df.csv"
    )

    process_dataset(
        bows_dataset,
        metadata,
        dataset_name,
        "conclusion_generation_belief_generation_valid_alshomary21.json",
        data_path / "valid_with_claim_df.csv"
    )

    metadata.add_evaluation_metric("f1_macro")
    metadata.write_metadata()
