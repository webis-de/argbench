from common import Genres, Output, Subareas, datasets_path, read_tabular, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import uuid

def process_dataset(bows_dataset, metadata, dataset_name, dataset_file, dataset_path):
    dataset = read_tabular(dataset_path)
    dataset["parsed_issues"] = dataset["big_issues"].str.findall("\d")
    output = Output(dataset_name)
    big_issues_definition = "; ".join(bows_dataset["topic"])
    output.append_definition(f"Given discussion topic and claim made by user, judge stance of the user to following big issues: {big_issues_definition}. " +
                             "Stance of user to big issue can be: Pro if positive, Con if negative and N/A if no stance can be deduced." +
                             "Stances should be in following format: [Big issue]: Pro\\n[Big issue]: Con\\n[Big issue]: N/A.")

    for row in dataset.iterrows():
        row = row[1]
        big_issue_labels = []
        for big_issue, rel in zip(bows_dataset["topic"], row["parsed_issues"]):
            if rel == "1":
                stance = "Pro"
            elif rel == "3":
                stance = "Con"
            else:
                stance = "N/A"
            issue_stance = f"{big_issue}: {stance}"
            big_issue_labels.append(issue_stance)

        label = "\n".join(big_issue_labels)
        prompt = f"Topic: {row['topic']}\nUser claim: {row['opinion_txt']}"
        id = str(uuid.uuid4())
        output.append_instance(id, prompt, [label])

    output.append_genre(Genres.DEBATE_PORTALS)
    output.append_subarea(Subareas.MINING)
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

    dataset_name = "stance_classification_belief_arguments_alshomary21"

    bows_dataset = read_tabular(bows_path / "big_issues_new_bows.csv")

    metadata = Metadata(dataset_name)

    process_dataset(
        bows_dataset,
        metadata,
        dataset_name,
        "stance_classification_belief_arguments_train_alshomary21.json",
        data_path / "train_with_claim_df.csv"
    )

    process_dataset(
        bows_dataset,
        metadata,
        dataset_name,
        "stance_classification_belief_arguments_test_alshomary21.json",
        data_path / "test_with_claim_df.csv"
    )

    process_dataset(
        bows_dataset,
        metadata,
        dataset_name,
        "stance_classification_belief_arguments_valid_alshomary21.json",
        data_path / "valid_with_claim_df.csv"
    )

    metadata.add_genre(Genres.DEBATE_PORTALS)
    metadata.add_skill(Subareas.MINING)
    
    metadata.write_metadata()
