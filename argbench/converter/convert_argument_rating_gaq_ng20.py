from common import Output, datasets_path, read_tabular, tasks_path, Metadata, add_seed_arg, set_seed, Genres, Skills
from argparse import ArgumentParser
import pandas as pd
cogency_description = "An argument is cogent if it has acceptable premises that are relevant to its conclusion and that are sufficient to draw the conclusion."
effectiveness_description = "An argument is effective if it persuades the target audience of (or corroborates agreement with) the author’s stance on the issue."
reasonableness_description = "An argument is reasonable if it contributes to the issue’s resolution in a sufficient way that is acceptable to the target audience."
overall_description = "How good an argument is holistically."

dataset_name = "argument_rating_gaq_ng20"
dataset_file_name_template =  "argument_rating_gaq_{split}_ng20.json"

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

def format_dataset(dataset, dimension_definitions,  metadata, split):
    task_output = Output(dataset_name)
    task_defintion = """Judge the quality of the following argument according to the given quality aspect. Possible outputs:
    "Very Low, Low, Medium, High, Very High, Cannot Judge."""
    dataset_file_name = dataset_file_name_template.format( split=split)
    task_output.append_definition(task_defintion)

    for row in dataset.iterrows():
        row = row[1]
        for dimension in dimension_definitions:
            dimension_definition = dimension_definitions[dimension]
            print(row[f"{dimension}_mean"])
            print(dimension)
            score = number_to_label(float(row[f"{dimension}_mean"]))
            prompt = f"Title: {row['title']}\nArgument: {row['text']}\n Quality Aspect: {dimension}\n Quality Aspect Definition: {dimension_definition}"
            id = row["id"]
            task_output.append_instance(id, prompt, [score])

    task_output.write_output(dataset_file_name)
    task_output.append_genre(Genres.WEB)
    task_output.append_subarea(Skills.QUALITY_ASSESSMENT)
    metadata.add_dataset(dataset_file_name, split)


if __name__ == "__main__":
    # Input arguments for dataset generation
    arg_parser = ArgumentParser()
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]


    data_path = datasets_path() / "gaq" / "all" # path to data

    # Set name of the dataset to identify it and files of that dataset


    # Class for collecting dataset file data
    # Dataset name specifies folder where dataset will be written
    dimension_defnitions = {
        "cogency": cogency_description,
        "effectiveness": effectiveness_description,
        "reasonableness" :reasonableness_description,
        "overall": overall_description
    }
    df_debate_crowd = convert_dataset(data_path / "debate_forums_crowd.csv")
    df_debate_experts = convert_dataset(data_path / "debate_forums_experts.csv")
    df_qa_crowd = convert_dataset(data_path / "qa_forums_crowd.csv")
    df_qa_experts = convert_dataset(data_path / "qa_forums_experts.csv")
    df_review_crowd = convert_dataset(data_path / "review_forums_crowd.csv")
    df_review_experts = convert_dataset(data_path / "review_forums_experts.csv")
    df_all = pd.concat([df_qa_crowd, df_qa_experts, df_debate_crowd, df_debate_crowd, df_review_crowd, df_review_crowd])

    df_test = df_all.sample(frac=0.2)
    df_training = df_all[~df_all["id"].isin(df_test["id"])]

    df_val = df_training.sample(frac=0.25)
    df_training = df_training[~df_training["id"].isin(df_val["id"])]


    metadata = Metadata(dataset_name)
    metadata.add_genre(Genres.WEB)
    metadata.add_skill(Skills.QUALITY_ASSESSMENT)

    format_dataset(df_training, dimension_defnitions,  metadata, "train")
    format_dataset(df_val, dimension_defnitions,  metadata, "val")
    format_dataset(df_test, dimension_defnitions, metadata, "test")
    metadata.add_evaluation_metric("fscore")
    metadata.write_metadata()