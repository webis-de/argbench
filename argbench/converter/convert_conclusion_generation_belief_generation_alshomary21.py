from common import Genres, Output, Skills, datasets_path, read_tabular, tasks_path, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import ast
import uuid

def process_dataset(bows_dataset, metadata, dataset_name, dataset_file, dataset_pathes, split):
    output = Output(dataset_name)
    output.append_definition("Given discussion topic and a collection of topic staces that describe users stance on various issues, generate a claim that is based user stance. Do not explain.")

    for dataset_path in dataset_pathes:
        dataset = read_tabular(dataset_path)
        dataset = dataset.dropna(axis=0)
        dataset["parsed_issues"] = dataset["big_issues"].str.findall("\d")

        for row in dataset.iterrows():
            row = row[1]
            total_topic_rel = []
            for i, issue_label in enumerate(row["parsed_issues"]):
                if issue_label == "1":
                    topic_str = f"{bows_dataset['topic'][i]}: Con"
                    total_topic_rel.append(topic_str)
                if issue_label == "3":
                    topic_str = f"{bows_dataset['topic'][i]}: Pro"
                    total_topic_rel.append(topic_str)
    #            else:
    #                topic_str = f"{bows_dataset['topic'][i]}: N/A"
    #                total_topic_rel.append(topic_str)

            total_topic_rel = "; ".join(total_topic_rel)
            prompt = f"Topic: {row['topic']}\nTopic Stances: {total_topic_rel}"
            label = row["top_claim"]
            id = str(uuid.uuid4())
            output.append_instance(id, prompt, [label])
        print(dataset_path)
        print(len(output.instances))
    output.append_genre(Genres.DEBATE_PORTALS)
    output.append_subarea(Skills.GENERATION)
    output.write_output(dataset_file)
    metadata.add_dataset(dataset_file, split)


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="What dataset will be processed?")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args) # Seed random number generation

    data_path = datasets_path() / "belief-arguments" / "data" # path to data
    dataset_name = "conclusion_generation_belief_generation_alshomary21"

    bows_dataset = read_tabular(data_path / "big_issues_bows" / "big_issues_new_bows.csv")
    bows_dataset = bows_dataset.dropna(axis=0)
    bows_dataset["pro_parsed"] = bows_dataset["pro_bow"].apply(ast.literal_eval)
    bows_dataset["con_parsed"] = bows_dataset["con_bow"].apply(ast.literal_eval)

    metadata = Metadata(dataset_name)

    process_dataset(
        bows_dataset,
        metadata,
        dataset_name,
        "conclusion_generation_belief_generation_train_alshomary21.json",
        [data_path / "preprocessed_data" / "train_with_claim_df.csv", data_path / "preprocessed_data" / "valid_with_claim_df.csv"],
        "train"
    )

    process_dataset(
        bows_dataset,
        metadata,
        dataset_name,
        "conclusion_generation_belief_generation_test_alshomary21.json",
        [data_path / "preprocessed_data" / "test_with_claim_df.csv"],
        "test"
    )

    metadata.add_genre(Genres.DEBATE_PORTALS)
    metadata.add_skill(Skills.GENERATION)
    metadata.add_evaluation_metric("generation-score")
    metadata.write_metadata()
