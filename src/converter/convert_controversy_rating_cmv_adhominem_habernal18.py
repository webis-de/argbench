import math

from common import Genres, Output, datasets_path, Metadata, add_seed_arg, set_seed, Skills
from argparse import ArgumentParser
import ndjson
import json
import uuid
from random import sample
# Labels end at 2.0 for some reason????
CONTROVERSY_MAPPING = {
    1: "Not Really Controversial",
    2: "Somehow Controversial",
    3: "Very Controversial"
}

REASONABLENESS_MAPPING = {
    1: "Quite Stupid",
    2: "Neutral",
    3: "Quite Reasonable"
}



reasonableness_dataset = "reasonableness_scoring_cmv_habernal18"
controversy_dataset = "controversy_scoring_cmv_habernal18"

controversy_dataset_template = "controversy_scoring_cmv_{split}_habernal18.json"
reaonsableness_dataset_template = "reasonableness_scoring_{split}_habernal18.json"

controversy_task_definition = "Classify the following post according to its controversy into either Not Really Controversial, Somehow Controversial, or Very Controversial. Do not explain."
reasonableness_task_definition = "Classify the following post according to its reasonableness into : Quite Stupid, Neutral, Quite Reasonable. Do not explain."

def process_dataset(data, split, controversy_metadata, reasonableness_metadata):
    controversy_data_file = controversy_dataset_template.format(split=split)
    reasonableness_data_file = reaonsableness_dataset_template.format(split=split)

    controversy_output = Output(controversy_dataset)
    controversy_output.append_definition(controversy_task_definition)

    reasonableness_output = Output(reasonableness_dataset)
    reasonableness_output.append_definition(reasonableness_task_definition)




    with open(controversy_path, "r") as cf:
        controversiality_data = json.load(cf)

    with open(stupidity_path, "r") as sf:
        reaonsablness_data = json.load(sf)

    for post in data:
        id = str(uuid.uuid4())
        controversy_score = controversiality_data[post["name"]]
        reasonableness_score = reaonsablness_data[post["name"]]
        controversy_idx = min([c for c in CONTROVERSY_MAPPING if controversy_score < c])
        reasonableness_idx = min([c for c in REASONABLENESS_MAPPING if reasonableness_score < c])
        controversy_label = CONTROVERSY_MAPPING[controversy_idx]
        reasonableness_label = REASONABLENESS_MAPPING[reasonableness_idx]
        prompt = f"Post: {post['body']}"

        controversy_output.append_instance(id, prompt, [controversy_label])
        reasonableness_output.append_instance(id, prompt, [reasonableness_label])


    controversy_output.append_genre(Genres.WEB_FORUMS)
    controversy_output.append_subarea(Skills.QUALITY_ASSESSMENT)
    controversy_output.write_output(controversy_data_file)


    reasonableness_output.append_genre(Genres.WEB_FORUMS)
    reasonableness_output.append_subarea(Skills.QUALITY_ASSESSMENT)
    reasonableness_output.write_output(reasonableness_data_file)


    controversy_metadata.add_dataset(controversy_data_file, split)
    controversy_metadata.add_skill(Skills.QUALITY_ASSESSMENT)
    controversy_metadata.add_genre(Genres.WEB_FORUMS)
    controversy_metadata.add_evaluation_metric("fscore")

    reasonableness_metadata.add_dataset(reasonableness_data_file, split)
    reasonableness_metadata.add_skill(Skills.QUALITY_ASSESSMENT)
    reasonableness_metadata.add_genre(Genres.WEB_FORUMS)
    controversy_metadata.add_evaluation_metric("fscore")

if __name__ == "__main__":
    # Input arguments for dataset generation
    arg_parser = ArgumentParser()
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args) # Seed random number generation

    data_path = datasets_path() / "cmv-adhominem" / "exported-1800-sampled-balanced-ops.json"
    controversy_path =  datasets_path() / "cmv-adhominem" / "annotated-1800-sampled-balanced-ops-controversy.json"
    stupidity_path = datasets_path() / "cmv-adhominem" / "annotated-1800-sampled-balanced-ops-stupidity.json"

    with open(data_path, "r") as df:
        post_data = ndjson.load(df)

    indices = range(len(post_data))
    test_size = math.ceil(len(post_data) * 0.2)
    test_indices = sample(indices, test_size )
    train_indices = [i for i in range(len(post_data)) if i not in test_indices]
    test_data = [post_data[i] for i in test_indices]
    train_data = [post_data[i] for i in train_indices]
    controversy_metadata = Metadata(controversy_dataset)
    reasonableness_metadata = Metadata(reasonableness_dataset)
    process_dataset(test_data, "test", controversy_metadata, reasonableness_metadata)
    process_dataset(train_data, "train", controversy_metadata, reasonableness_metadata)
    controversy_metadata.write_metadata()
    reasonableness_metadata.write_metadata()