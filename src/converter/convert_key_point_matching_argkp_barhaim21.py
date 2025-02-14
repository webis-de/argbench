from common import Output, read_tabular, datasets_path, tasks_path, Metadata, add_seed_arg, set_seed, Genres, Skills, find_topic_size_to_split
from argparse import ArgumentParser
import uuid

dataset_name = "key_point_matching_argkp_barhaim21"
dataset_file_template = "key_point_matching_argkp_{split}_barhaim21.json"


def preprocess_data(dataset, split, metadata):
    output = Output(dataset_name)

    dataset_file = dataset_file_template.format(split=split)

    output.append_definition("""Judge if the following keypoint summarizes the given argument.
     Possible responses: Match if argument is summarized by key point and No Match if argument is not summarized by key point. Do not explain""")
    for row in dataset.iterrows():
        row = row[1]
        prompt = f"Argument: {row['argument']}\nKey Point: {row['key_point']}"
        response = "Match" if row["label"] == 1 else "No Match"
        wrong_response = "No Match" if row["label"] == 1 else "Match"
        id = str(uuid.uuid4())
        output.append_positive_example(prompt, response, "")
        output.append_negative_example(prompt, wrong_response, "")

        output.append_instance(id, prompt, [response])
    metadata.add_evaluation_metric("fscore")
    metadata.add_dataset(dataset_file)
    metadata.add_genre(Genres.WIKIPEDIA)
    metadata.add_skill(Skills.PERSPECTIVE_ASSESSMENT)
    output.append_subarea(Skills.PERSPECTIVE_ASSESSMENT)
    output.append_genre(Genres.WIKIPEDIA)
    output.write_output(dataset_file)


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Program to convert ajjour unit segmentation dataset into appropriate form")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    dataset_path = str(datasets_path()
                    / "argkpa-2021"
                    / "ArgKP-2021_dataset.csv")

    dataset = read_tabular(dataset_path)
    df_test, df_train = find_topic_size_to_split(dataset, "topic")

    metadata = Metadata(dataset_name)
    preprocess_data(df_test, "test", metadata)
    preprocess_data(df_train, "train", metadata)


    metadata.write_metadata()
