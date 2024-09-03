from common import Output, datasets_path, read_tabular, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser

dataset_name = "warrant_identification_semeval_2018_task_12_habernal18"


def process_split(dataset, metadata, split):
    output = Output(dataset_name)
    dataset_file = f"warrant_identification_semeval_2018_task_12_{split}_habernal18.json"
    output.append_definition("""Given the following reason and claim along with the debate title and a short description of the debate they occur in, 
                             identify the correct warrant from two candidates\n
                             Warrant 1 and Warrant 2. The warrant explains why the claim follows from the reason. """)

    for row in dataset.iterrows():
        row = row[1]
        id = row["#id"]
        warrant_1 = row["warrant0"]
        warrant_2 = row["warrant1"]
        title = row["debateTitle"]
        description = row["debateInfo"]
        reason = row["reason"]
        claim = row["claim"]
        label = f"Warrant 1: {warrant_1}." if row["correctLabelW0orW1"] else f"Warrant 2: {warrant_2}."
        prompt = f"Debate Title: {title}\nDebate Description: {description}\nReason: {reason}.\nWarrant 1: {warrant_1}.\n Warrant 2: {warrant_2},\nClaim: {claim}."
        output.append_instance(id, prompt, [label])

    metadata.add_dataset(dataset_file, split)
    output.write_output(dataset_file)


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="What dataset will be processed?")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    data_folder = datasets_path() / "semeval-18-task12" / "data"

    metadata = Metadata(dataset_name)

    train_dataset = read_tabular(data_folder / "preprocessed_trn.txt", separator="\t")
    test_dataset = read_tabular(data_folder / "preprocessed_tst.txt", separator="\t")
    dev_dataset = read_tabular(data_folder / "preprocessed_dev.txt", separator="\t")

    print("Train")
    process_split(train_dataset, metadata, "train")
    print("Test")
    process_split(test_dataset, metadata, "test")
    print("Dev")
    process_split(dev_dataset, metadata, "dev")

    metadata.add_evaluation_metric("f1_macro")
    metadata.write_metadata()
