from common import Output, datasets_path, read_tabular, tasks_path, Metadata, add_seed_arg, set_seed, Genres, Subareas
from argparse import ArgumentParser

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


def convert_dataset(data_path, dataset_name, data_name, metadata):
    dataset = read_tabular(data_path)

    cogency_output = Output(dataset_name)
    effectiveness_output = Output(dataset_name)
    reasonableness_output = Output(dataset_name)

    cogency_dataset = data_name.format("cogency")
    effectiveness_dataset = data_name.format("effectiveness")
    reasonableness_dataset = data_name.format("reasonableness")

    cogency_output.append_definition("Judge the quality of text arguments according to quality aspect: cogency. " +
                                     f"Quality aspect description: {cogency_description} Possible outputs: " +
                                     "Very Low, Low, Medium, High, Very High, Cannot Judge.")
    effectiveness_output.append_definition("Judge the quality text arguments according to quality aspect: effectiveness. " +
                                     f"Quality aspect description: {effectiveness_description} Possible outputs: " +
                                     "Very Low, Low, Medium, High, Very High, Cannot Judge.")
    reasonableness_output.append_definition("Judge the quality of text arguments according to quality aspect: reasonableness. " +
                                     f"Quality aspect description: {reasonableness_description} Possible outputs: " +
                                     "Very Low, Low, Medium, High, Very High, Cannot Judge.")

    dataset[dataset["cogency_mean"] == "#"] = 5
    dataset[dataset["effectiveness_mean"] == "#"] = 5
    dataset[dataset["reasonableness_mean"] == "#"] = 5
    dataset["cogency_mean"] = dataset["cogency_mean"].astype("float")
    dataset["effectiveness_mean"] = dataset["effectiveness_mean"].astype("float")
    dataset["reasonableness_mean"] = dataset["reasonableness_mean"].astype("float")

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

    convert_dataset(data_path / "debate_forums_crowd.csv", dataset_name, "argument_ranking_debate_forums_crowd_{}_ng20.json", metadata)
    convert_dataset(data_path / "debate_forums_experts.csv", dataset_name, "argument_ranking_debate_forums_experts_{}_ng20.json", metadata)
    convert_dataset(data_path / "qa_forums_crowd.csv", dataset_name, "argument_ranking_qa_forums_crowd_{}_ng20.json", metadata)
    convert_dataset(data_path / "qa_forums_experts.csv", dataset_name, "argument_ranking_qa_forums_experts_{}_ng20.json", metadata)
    convert_dataset(data_path / "review_forums_crowd.csv", dataset_name, "argument_ranking_review_forums_crowd_{}_ng20.json", metadata)
    convert_dataset(data_path / "review_forums_experts.csv", dataset_name, "argument_ranking_review_forums_experts_{}_ng20.json", metadata)

    metadata.add_genre(Genres.DEBATE_PORTALS)
    metadata.add_subarea(Subareas.RANKING)
    metadata.write_metadata()
