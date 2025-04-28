from argbench.converter.common import find_topic_size_to_split
from common import Genres, Output, Skills, datasets_path, read_tabular, tasks_path, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import uuid

DATASET_NAME = "stance_classification_ibmsc_barhaim17"


def process_dataset(dataset, path):
    output = Output(DATASET_NAME)
    print(f"processing {path}")
    output.append_definition("""Classify the stance of the following claim into Pro or Con. Answer with Pro if the following claim supports the following topic. Answer with Con if the claim attacks the topic.
    Only answer with Pro or Con.""")

    for row in dataset.iterrows():
        row = row[1]
        prompt = f"Topic: {row['topicText']}\nClaim: {row['claims.claimCorrectedText']}"
        model_output = "Pro" if row["claims.stance"] == "PRO" else "Con"
        wrong_output = "Con" if row["claims.stance"] == "PRO" else "Pro"
        id = str(uuid.uuid4())
        output.append_positive_example(prompt, model_output, "")
        output.append_negative_example(prompt, wrong_output, "")
        output.append_instance(id, prompt, [model_output])

    output.append_genre(Genres.WIKIPEDIA)
    output.append_subarea(Skills.PERSPECTIVE_ASSESSMENT)
    output.write_output(path)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    metadata = Metadata(DATASET_NAME)

    dataset = str(datasets_path() / "ibmsc" / "ibmsc-stance-classification.csv")

    data = read_tabular(dataset)
    df_train = data[data["split"] == "train"]

    df_val, df_train = find_topic_size_to_split(df_train, "topicText", 0.25)

    process_dataset(df_train, "stance_classification_ibmsc_train_barhaim17.json")
    process_dataset(df_val, "stance_classification_ibmsc_val_barhaim17.json")
    process_dataset(data[data["split"] == "test"], "stance_classification_ibmsc_test_barhaim17.json")


    metadata.add_dataset("stance_classification_ibmsc_train_barhaim17.json", "train")
    metadata.add_dataset("stance_classification_ibmsc_test_barhaim17.json", "test")
    metadata.add_dataset("stance_classification_ibmsc_val_barhaim17.json", "val")

    metadata.add_genre(Genres.WIKIPEDIA)
    metadata.add_skill(Skills.PERSPECTIVE_ASSESSMENT)
    metadata.add_evaluation_metric("fscore")

    metadata.write_metadata()
