from common import Output, datasets_path, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import ndjson
import json
import uuid

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

if __name__ == "__main__":
    # Input arguments for dataset generation
    arg_parser = ArgumentParser(description="What dataset will be processed?")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args) # Seed random number generation

    data_path = datasets_path() / "argument-detection" / "habernal18-ad-hominem-detection" / "exported-1800-sampled-balanced-ops.json"
    controversy_path =  datasets_path() / "argument-detection" / "habernal18-ad-hominem-detection" / "annotated-1800-sampled-balanced-ops-controversy.json"
    stupidity_path = datasets_path() / "argument-detection" / "habernal18-ad-hominem-detection" / "annotated-1800-sampled-balanced-ops-stupidity.json"

    dataset_name = "fallacy_detection_cmv_adhominem_habernal18"
    controversy_dataset_file = "post_controversy_cmv_adhominem_habernal18.json"

    output = Output(dataset_name)
    output.append_definition("Classify post contents according to controversy and reasonableness. Valid labels for controversy: Not Really Controversial, Somehow Controversial, Very Controversial. " +
                             "Valid labels for reasonableness: Quite Stupid, Neutral, Quite Reasonable. " +
                             "Output should be formatted in the form: {controversy_label};{reasonableness_label} and nothing else.")

    metadata = Metadata(dataset_name)

    with open(data_path, "r") as df:
        post_data = ndjson.load(df)

    with open(controversy_path, "r") as cf:
        controversiality_data = json.load(cf)

    with open(stupidity_path, "r") as sf:
        stupidity_data = json.load(sf)

    for post in post_data:
        id = str(uuid.uuid4())
        controversy_score = controversiality_data[post["name"]]
        reasonableness_score = stupidity_data[post["name"]]
        controversy_idx = min([c for c in CONTROVERSY_MAPPING if controversy_score < c])
        reasonableness_idx = min([c for c in REASONABLENESS_MAPPING if reasonableness_score < c])
        controversy_label = CONTROVERSY_MAPPING[controversy_idx]
        reasonableness_label = REASONABLENESS_MAPPING[reasonableness_idx]
        prompt = f"Post Text: {post['body']}"
        label = f"{controversy_label};{reasonableness_label}"
        output.append_instance(id, prompt, [label])

    output.write_output(controversy_dataset_file)
    metadata.add_dataset(controversy_dataset_file)
    metadata.add_evaluation_metric("f1_macro")
    metadata.write_metadata()

