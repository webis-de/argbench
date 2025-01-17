from common import Genres, Output, Subareas, datasets_path, read_tabular, tasks_path, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import uuid

DATASET_NAME = "stance_classification_ibmsc_barhaim17"


def process_dataset(dataset, path):
    output = Output(DATASET_NAME)

    output.append_definition("Classify the stance of the following claim into Pro or Con. Output Pro if the claim supports the topic and output Con if the claim attacks the topic. Only answer with Con or Pro.")

    for row in dataset.iterrows():
        row = row[1]
        prompt = f"Topic: {row['topicText']}\nClaim: {row['claims.claimCorrectedText']}"
        model_output = "Pro" if row["claims.stance"] == "PRO" else "Con"
        wrong_output = "Con" if row["claims.stance"] == "PRO" else "Pro"
        id = str(uuid.uuid4())
        output.append_positive_example(prompt, model_output, "")
        output.append_negative_example(prompt, wrong_output, "")
        output.append_instance(id, prompt, [model_output])

    output.append_genre(Genres.DEBATES)
    output.append_subarea(Subareas.MINING)
    output.write_output(path)


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Program to convert gretz20 ibm quality dataset into appropriate form")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    metadata = Metadata(DATASET_NAME)

    dataset = str(datasets_path() / "ibmsc" / "ibmsc-stance-classification.csv")

    data = read_tabular(dataset)

    process_dataset(data[data["split"] == "train"], "stance_classification_ibmsc_train_barhaim17.json")
    process_dataset(data[data["split"] == "test"], "stance_classification_ibmsc_test_barhaim17.json")


    metadata.add_dataset("stance_classification_ibmsc_train_barhaim17.json", "train")
    metadata.add_dataset("stance_classification_ibmsc_test_barhaim17.json", "test")

    metadata.add_genre(Genres.DEBATES)
    metadata.add_subarea(Subareas.MINING)
    metadata.add_evaluation_metric("f1_macro")

    metadata.write_metadata()
