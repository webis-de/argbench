import math

from common import Genres, Output, Skills, datasets_path, Metadata, add_seed_arg, set_seed, split_test_val_train
from argparse import ArgumentParser
import ndjson
from random import sample

dataset_name = "fallacy_detection_cmv_adhominem_habernal18"
dataset_file_format = "fallacy_detection_cmv_adhominem_{split}_habernal18.json"


def process_data(data, split):

    dataset_file_name = dataset_file_format.format(split=split)
    output = Output(dataset_name)
    output.append_definition("Classify if the following argument is an ad-hominem (personal attack) or not. Answer with Ad-hominem or Not-ad-hominem.")

    for post in data:

        if post["violated_rule"] == 2:
            ad_hominem = "Ad-hominem"
        else:
            ad_hominem = "Not-ad-hominem"
        argument = post["body"]
        output.append_instance(id=post["id"],input=f"Argument: {argument}", output=[ad_hominem])

    output.append_genre(Genres.WEB_FORUMS)
    output.append_subarea(Skills.REASONING)
    output.write_output(dataset_file_name)
    metadata.add_dataset(dataset_file_name, split)
    metadata.add_genre(Genres.WEB_FORUMS)
    metadata.add_skill(Skills.REASONING)
    metadata.add_evaluation_metric("fscore")
    metadata.write_metadata()


if __name__ == "__main__":
    # Input arguments for dataset generation
    arg_parser = ArgumentParser()
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
     # Seed random number generation

    data_path = datasets_path() / "cmv-adhominem"

    fallacy_file = data_path / "exported-3621-sampled-positive-negative-ah-no-context.json"

    metadata = Metadata(dataset_name)
    with open(fallacy_file, "r") as df:
        post_data = ndjson.load(df)
    test_data, val_data, train_data = split_test_val_train(post_data)

    process_data(test_data, "test")
    process_data(val_data, "val")
    process_data(train_data, "train")