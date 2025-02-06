from common import Output, datasets_path, read_tabular, tasks_path, Metadata, add_seed_arg, set_seed, Genres, Subareas
from argparse import ArgumentParser
import pandas as pd
cogency_description = "An argument is cogent if it has acceptable premises that are relevant to its conclusion and that are sufficient to draw the conclusion."
effectiveness_description = "Argumentation is effective if it persuades the target audience of (or corroborates agreement with) the author’s stance on the issue."
reasonableness_description = "Argumentation is reasonable if it contributes to the issue’s resolution in a sufficient way that is acceptable to the target audience."


def number_to_label(number):
    if number < 2:
        return "Very Low"
    if number < 3:
        return "Low"
    if number < 4:
        return "Medium"
    if number < 5:
        return "Very High"
    return "Cannot Judge"


def convert_dataset(data_path):
    dataset = read_tabular(data_path)



    dataset[dataset["cogency_mean"] == "#"] = 5
    dataset[dataset["effectiveness_mean"] == "#"] = 5
    dataset[dataset["reasonableness_mean"] == "#"] = 5
    dataset["cogency_mean"] = dataset["cogency_mean"].astype("float")
    dataset["effectiveness_mean"] = dataset["effectiveness_mean"].astype("float")
    dataset["reasonableness_mean"] = dataset["reasonableness_mean"].astype("float")
    return dataset

def format_dataset(dataset, dataset_name, data_name, metadata):

    cogency_output = Output(dataset_name)
    effectiveness_output = Output(dataset_name)
    reasonableness_output = Output(dataset_name)
    cogency_output.append_definition("Judge the quality of text arguments according to quality aspect: cogency. " +
                                     f"Quality aspect description: {cogency_description} Possible outputs: " +
                                     "Very Low, Low, Medium, High, Very High, Cannot Judge.")
    effectiveness_output.append_definition("Judge the quality text arguments according to quality aspect: effectiveness. " +
                                           f"Quality aspect description: {effectiveness_description} Possible outputs: " +
                                           "Very Low, Low, Medium, High, Very High, Cannot Judge.")
    reasonableness_output.append_definition("Judge the quality of text arguments according to quality aspect: reasonableness. " +
                                            f"Quality aspect description: {reasonableness_description} Possible outputs: " +
                                            "Very Low, Low, Medium, High, Very High, Cannot Judge.")

    cogency_dataset = data_name.format("cogency")
    effectiveness_dataset = data_name.format("effectiveness")
    reasonableness_dataset = data_name.format("reasonableness")

    for row in dataset.iterrows():
        row = row[1]
        cogency = number_to_label(row["cogency_mean"])
        effectiveness = number_to_label(row["effectiveness_mean"])
        reasonableness = number_to_label(row["reasonableness_mean"])
        prompt = f"Title: {row['title']}\nText: {row['text']}"
        id = row["id"]
        cogency_output.append_instance(id, prompt, [cogency])
        effectiveness_output.append_instance(id, prompt, [effectiveness])
        reasonableness_output.append_instance(id, prompt, [reasonableness])

    cogency_output.write_output(cogency_dataset)
    cogency_output.append_genre(Genres.DEBATE_PORTALS)
    cogency_output.append_subarea(Subareas.RANKING)
    effectiveness_output.write_output(effectiveness_dataset)
    effectiveness_output.append_genre(Genres.DEBATE_PORTALS)
    effectiveness_output.append_subarea(Subareas.RANKING)
    reasonableness_output.write_output(reasonableness_dataset)
    reasonableness_output.append_genre(Genres.DEBATE_PORTALS)
    reasonableness_output.append_subarea(Subareas.RANKING)
    metadata.add_dataset(cogency_dataset)
    metadata.add_dataset(effectiveness_dataset)
    metadata.add_dataset(reasonableness_dataset)


if __name__ == "__main__":
    # Input arguments for dataset generation
    arg_parser = ArgumentParser(description="What dataset will be processed?")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args) # Seed random number generation

    data_path = datasets_path() / "gaq" / "all" # path to data

    # Set name of the dataset to identify it and files of that dataset
    dataset_name = "argument_ranking_gaq_ng20"

    # Class for collecting dataset file data
    # Dataset name specifies folder where dataset will be written

    metadata = Metadata(dataset_name)
    metadata.add_evaluation_metric("f1_macro")

    df_debate_crowd = convert_dataset(data_path / "debate_forums_crowd.csv")
    df_debate_experts = convert_dataset(data_path / "debate_forums_experts.csv")
    df_qa_crowd = convert_dataset(data_path / "qa_forums_crowd.csv")
    df_qa_experts = convert_dataset(data_path / "qa_forums_experts.csv")
    df_review_crowd = convert_dataset(data_path / "review_forums_crowd.csv")
    df_review_experts = convert_dataset(data_path / "review_forums_experts.csv")
    df_all = pd.concat([df_qa_crowd, df_qa_experts, df_debate_crowd, df_debate_crowd, df_review_crowd, df_review_crowd])

    df_test = df_all.sample(frac=0.2)
    df_training = df_all[~df_all["id"].isin(df_test["id"])]

    metadata.add_genre(Genres.DEBATE_PORTALS)
    metadata.add_subarea(Subareas.RANKING)
    metadata.write_metadata()

    format_dataset(df_training, dataset_name, "argument_ranking_{}_train_ng20.json", metadata)
    format_dataset(df_test, dataset_name, "argument_ranking_{}_test_ng20.json", metadata)
