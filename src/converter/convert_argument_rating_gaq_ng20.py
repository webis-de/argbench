from common import Output, datasets_path, read_tabular, tasks_path, Metadata, add_seed_arg, set_seed, Genres, Subareas
from argparse import ArgumentParser
import pandas as pd
cogency_description = "An argument is cogent if it has acceptable premises that are relevant to its conclusion and that are sufficient to draw the conclusion."
effectiveness_description = "An argument is effective if it persuades the target audience of (or corroborates agreement with) the author’s stance on the issue."
reasonableness_description = "An argument is reasonable if it contributes to the issue’s resolution in a sufficient way that is acceptable to the target audience."

dataset_name_template = "argument_rating_{dimension}_ng20"
dataset_file_name_template =  "argument_rating_{dimension}_{split}_ng20.json"

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

def format_dataset(dataset, dimension_definition, dimension,  metadata, split):


    dataset_name = dataset_name_template.format(dimension=dimension)
    dataset_file_name = dataset_file_name_template.format(dimension=dimension, split=split)
    task_output = Output(dataset_name)
    task_output.append_definition(dimension_definition)

    for row in dataset.iterrows():
        row = row[1]
        score = number_to_label(row[f"{dimension}_mean"])
        prompt = f"Title: {row['title']}\nText: {row['text']}"
        id = row["id"]
        task_output.append_instance(id, prompt, [score])

    task_output.write_output(dataset_file_name)
    task_output.append_genre(Genres.DEBATE_PORTALS)
    task_output.append_subarea(Subareas.RANKING)
    metadata.add_dataset(dataset_file_name, split)


if __name__ == "__main__":
    # Input arguments for dataset generation
    arg_parser = ArgumentParser(description="What dataset will be processed?")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args) # Seed random number generation

    data_path = datasets_path() / "gaq" / "all" # path to data

    # Set name of the dataset to identify it and files of that dataset


    # Class for collecting dataset file data
    # Dataset name specifies folder where dataset will be written
    task_defnitions = {
        "cogency": f"""Judge the quality of the following argument according to quality aspect: cogency.  
                                     "Quality aspect description: {cogency_description} Possible outputs: 
                                     "Very Low, Low, Medium, High, Very High, Cannot Judge."""
                   ,
        "effectiveness": f"""Judge the quality the following argument according to quality aspect: effectiveness. 
                                           Quality aspect description: {effectiveness_description} Possible outputs: 
                                           Very Low, Low, Medium, High, Very High, Cannot Judge.""",

        "reasonableness" : f"""Judge the quality of the following argument according to quality aspect: reasonableness. 
                                            Quality aspect description: {reasonableness_description} Possible outputs: 
                                            Very Low, Low, Medium, High, Very High, Cannot Judge."""
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
    for dimension in task_defnitions:
        dataset_name = dataset_name_template.format(dimension=dimension)
        metadata = Metadata(dataset_name)
        metadata.add_genre(Genres.DEBATE_PORTALS)
        metadata.add_subarea(Subareas.RANKING)
        metadata.write_metadata()
        task_definition = task_defnitions[dimension]
        format_dataset(df_training, task_definition, dimension, metadata, "train")
        format_dataset(df_test, task_definition, dimension, metadata, "test")
