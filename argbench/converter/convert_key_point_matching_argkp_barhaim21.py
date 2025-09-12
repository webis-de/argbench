from common import Output, read_tabular, datasets_path, tasks_path, Metadata, add_seed_arg, set_seed, Genres, Skills, find_topic_size_to_split
from argparse import ArgumentParser
import uuid

dataset_name = "key_point_matching_argkp_barhaim21"
dataset_file_template = "key_point_matching_argkp_{split}_barhaim21.json"


def preprocess_data(dataset, split, metadata):
    output = Output(dataset_name)

    dataset_file = dataset_file_template.format(split=split)

    output.append_definition("""Judge if the following keypoint summarizes the given argument. A key point is a short talking point.
     Key points may be viewed as high-level arguments. They should be general enough to match a significant portion of the arguments, yet informative enough to make a useful summary.
     Possible responses: Match if argument is summarized by key point and No Match if argument is not summarized by key point.
     Only output Match or No-match.""")
    for row in dataset.iterrows():
        row = row[1]
        prompt = f"Key Point: {row['key_point']}\nArgument: {row['argument']}"
        response = "Match" if row["label"] == 1 else "No-match"
        wrong_response = "No-match" if row["label"] == 1 else "Match"
        id = str(uuid.uuid4())
        output.append_positive_example(prompt, response, "")
        output.append_negative_example(prompt, wrong_response, "")

        output.append_instance(id, prompt, [response])
    metadata.add_evaluation_metric("fscore")
    metadata.add_dataset(dataset_file, split)
    metadata.add_genre(Genres.WIKIPEDIA)
    metadata.add_skill(Skills.PERSPECTIVE_ASSESSMENT)
    output.append_subarea(Skills.PERSPECTIVE_ASSESSMENT)
    output.append_genre(Genres.WIKIPEDIA)
    output.write_output(dataset_file)


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Program to convert ajjour unit segmentation dataset into appropriate form")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    

    dataset_path = str(datasets_path()
                    / "argkpa-2021"
                    / "ArgKP-2021_dataset.csv")

    dataset = read_tabular(dataset_path)
    df_test, df_train = find_topic_size_to_split(dataset, "topic", 0.2)
    df_val, df_train = find_topic_size_to_split(df_train, "topic", 0.25)

    metadata = Metadata(dataset_name)
    preprocess_data(df_test, "test", metadata)
    preprocess_data(df_train, "train", metadata)
    preprocess_data(df_val, "val", metadata)

    metadata.write_metadata()
