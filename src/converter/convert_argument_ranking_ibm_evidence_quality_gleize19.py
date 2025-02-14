from common import Output, datasets_path, read_tabular, tasks_path, Metadata, add_seed_arg, set_seed, Genres, Skills
from argparse import ArgumentParser
import uuid


def convert_dataset(data_path, data_file, metadata, split):
    dataset = read_tabular(data_path)
    output = Output(dataset_name)
    output.append_definition("Given the topic and the two evidences, is the first evidence more convincing than" +
                             "the second evidence. Only respond with better or worse, do not explain.")

    dataset["label"] = dataset["label"].map({1: "Better", 2: "Worse"})

    for row in dataset.iterrows():
        row = row[1]
        topic = row["topic"]
        evidence_0 = row["evidence_1"]
        evidence_1 = row["evidence_2"]
        label = row["label"]
        prompt = f"Topic: {topic}\nEvidence 1: {evidence_0} \nEvidence 2: {evidence_1}"
        id = str(uuid.uuid4())
        output.append_instance(id, prompt, [label])

    output.write_output(data_file)
    output.append_genre(Genres.WIKIPEDIA)
    output.append_subarea(Skills.QUALITY_ASSESSMENT)
    metadata.add_dataset(data_file, split)


if __name__ == "__main__":
    # Input arguments for dataset generation
    arg_parser = ArgumentParser()
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args) # Seed random number generation

    data_path = datasets_path() / "ibm-evidence-quality" # path to data

    # Set name of the dataset to identify it and files of that dataset
    dataset_name = "argument_ranking_ibm_evidence_quality_gleize19"
    dataset_file_train = "argument_ranking_ibm_evidence_quality_train_gleize19.json"
    dataset_file_test = "argument_ranking_ibm_evidence_quality_test_gleize19.json"

    metadata = Metadata(dataset_name)

    convert_dataset(data_path / "train.csv", dataset_file_train, metadata, "train")
    convert_dataset(data_path / "test.csv", dataset_file_test, metadata, "test")
    metadata.add_evaluation_metric("fscore")
    metadata.add_genre(Genres.WIKIPEDIA)
    metadata.add_skill(Skills.QUALITY_ASSESSMENT)
    metadata.write_metadata()
