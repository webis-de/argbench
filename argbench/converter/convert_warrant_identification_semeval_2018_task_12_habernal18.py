import pandas as pd

from common import Genres, Output, Skills, datasets_path, read_tabular, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser

dataset_name = "warrant_identification_semeval_2018_task_12_habernal18"


def process_split(dataset, metadata, split):
    output = Output(dataset_name)
    dataset_file = f"warrant_identification_semeval_2018_task_12_{split}_habernal18.json"
    output.append_definition("""Given the following reason and claim along with the debate title and a short description of the debate they occur in, 
                             identify the correct warrant from two candidates\n
                             Warrant 1 and Warrant 2. The warrant explains why the claim follows from the reason. Only output Warrant 1 or Warrant 2.""")

    for row in dataset.iterrows():
        row = row[1]
        id = row["#id"]
        warrant_1 = row["warrant0"]
        warrant_2 = row["warrant1"]
        title = row["debateTitle"]
        description = row["debateInfo"]
        reason = row["reason"]
        claim = row["claim"]
        #label = f"Warrant 1: {warrant_1}." if row["correctLabelW0orW1"] else f"Warrant 2: {warrant_2}."
        label = f"Warrant 2" if row["correctLabelW0orW1"] else f"Warrant 1"
        prompt = f"Debate Title: {title}\nDebate Description: {description}\nReason: {reason}.\nWarrant 1: {warrant_1}.\n Warrant 2: {warrant_2},\nClaim: {claim}."
        output.append_instance(id, prompt, [label])

    metadata.add_dataset(dataset_file, split)
    output.append_genre(Genres.NEWS)
    output.append_subarea(Skills.REASONING)
    output.write_output(dataset_file)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    

    data_folder = datasets_path() / "semeval-18-task12" / "data"

    metadata = Metadata(dataset_name)

    train_dataset = read_tabular(data_folder / "preprocessed_trn.txt", separator="\t")
    test_dataset = read_tabular(data_folder / "preprocessed_tst.txt", separator="\t")
    dev_dataset = read_tabular(data_folder / "preprocessed_dev.txt", separator="\t")

    print(f"{len(train_dataset)}")
    print(f"{len(dev_dataset)}")


    print(f"{len(train_dataset)}")
    print("Train")
    process_split(train_dataset, metadata, "train")
    process_split(dev_dataset, metadata, "val")
    print("Test")
    process_split(test_dataset, metadata, "test")

    metadata.add_genre(Genres.NEWS)
    metadata.add_skill(Skills.REASONING)
    metadata.add_evaluation_metric("fscore")
    metadata.write_metadata()
