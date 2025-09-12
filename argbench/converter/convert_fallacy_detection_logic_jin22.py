from uuid import uuid4

import pandas as pd

from common import Output, datasets_path, read_tabular, tasks_path, Metadata, add_seed_arg, set_seed, Genres, Skills
from argparse import ArgumentParser

dataset_name = "fallacy_detection_logic_jin22"
map = {
    "intentional": "Intentional",
    "faulty generalization" : "Faulty Generalization",
    "ad hominem": "Ad-hominem",
    "fallacy of relevance" : "Fallacy of Relevance",
    "false causality": "False Causality",
    "appeal to emotion": "Appeal to Emotion",
    "fallacy of extension" : "Fallacy of Extension",
    "fallacy of logic" : "Fallacy of Logic",
    "fallacy of credibility" : "Fallacy of Credibility",
    "ad populum": "Ad-populum",
    "equivocation":"Equivocation",
    "circular reasoning": "Circular Reasoning",
    "false dilemma" : "False Dilemma"
}
def process_data(dataset, data_name, data_split, metadata):

    output = Output(dataset_name)
    output.append_definition("A fallacy a failure in reasoning which renders an argument invalid. Classify the following sentence into one of the following fallacies: " +

                             "Intentional: some intentional (sometimes subconscious) action/choice to incorrectly support an argument.\n" +
                             "Faulty Generalization: an informal fallacy wherein a conclusion is drawn about all or many instances of a phenomenon on the basis of one or a few instances of that phenomenon. is an example of jumping to conclusions.\n" +
                             "Ad-hominem: instead of addressing someone's argument or position, you irrelevantly attack the person or some aspect of the person who is making the argument.\n" +
                             "Fallacy of Relevance: introducing premises or conclusions that have nothing to do with the subject matter.\n" +
                             "False Causality: statement that jumps to a conclusion implying a causal relationship without supporting evidence.\n" +
                             "Appeal to Emotion: manipulation of the recipient's emotions in order to win an argument.\n" +
                             "Fallacy of Extension: attacking an exaggerated or caricatured version of your opponent's position.\n" +
                             "Fallacy of Logic: an error in the logical structure of an argument.\n" +
                             "Fallacy of Credibility: attempts to disprove an argument by attacking the character of the speaker.\n" +
                             "Ad-populum: a fallacious argument which is based on affirming that something is real or better because the majority thinks so\n" +
                             "Equivocation: when a key term or phrase in an argument is used in an ambiguous way, with one meaning in one portion of the argument and then another meaning in another portion of the argument.\n" +
                             "Circular Reasoning: when the end of an argument comes back to the beginning without having proven itself.\n" +
                             "False Dilemma: presenting only two options or sides when there are many options or sides.\n")

    for row in dataset.iterrows():
        row = row[1]
        id = str(uuid4())
        if len(row['source_article']) > 500:
            print(row['source_article'])
        prompt = f"Sentence: {row['source_article']}"
        if "logical_fallacies" in row:
            fallacies = row["logical_fallacies"]
        elif "updated_label" in row:
            fallacies = row["updated_label"]
        else:
            raise Exception("No labels found!")


        output.append_instance(id, prompt, [map[fallacies] ])


    output.append_genre(Genres.WEB)
    output.append_subarea(Skills.REASONING)
    output.write_output(data_name)
    metadata.add_dataset(data_name, data_split)
    metadata.add_skill(Skills.REASONING)
    metadata.add_genre(Genres.WEB)
    metadata.add_evaluation_metric("fscore")
    return len(output.instances)

def read_dataset(files):
    all_dfs = []
    for file in files:
        dataset = read_tabular(data_path / file)
        if "updated_label" in dataset.columns:
            dataset.rename(columns={"updated_label":"logical_fallacies"}, inplace=True)
            all_dfs.append(dataset[["source_article", "logical_fallacies"]])
    return pd.concat(all_dfs)

if __name__ == "__main__":
    # Input arguments for dataset generation
    arg_parser = ArgumentParser(description="What dataset will be processed?")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
#     # Seed random number generation
    data_path = datasets_path() / "logic" # path to data
    all_training_datasets = ["climate_train.csv", "edu_train.csv"]
    all_val_datasets = ["climate_dev.csv", "edu_dev.csv"]
    all_test_datasets = ["climate_test.csv", "edu_test.csv"]
    all_training_df = []
    df_train = read_dataset(all_training_datasets)
    df_val = read_dataset(all_val_datasets)
    df_test = read_dataset(all_test_datasets)

    metadata = Metadata(dataset_name)

    count_train = process_data(df_train, "fallacy_detection_logic_train_goffredo23.json", "train", metadata)
    count_test = process_data(df_test, "fallacy_detection_logic_test_goffredo23.json", "test", metadata)
    count_val = process_data(df_val, "fallacy_detection_logic_val_goffredo23.json", "val", metadata)

    print(f"Found {count_test + count_train + count_val} sentences")
    
    metadata.add_genre(Genres.WEB)
    metadata.add_skill(Skills.REASONING)
    metadata.write_metadata()
