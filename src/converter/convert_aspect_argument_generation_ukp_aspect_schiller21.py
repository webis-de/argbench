from common import Output, datasets_path, read_tabular, tasks_path, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import uuid
import ndjson
import random

DATASET_NAME = "aspect_argument_generation_ukp_aspect_schiller21"
DATASET_FILE = "aspect_argument_generation_ukp_aspect_schiller21.json"

def read_generated(path):
    con_path = path / "generated_training_data_CON_0.jsonl"
    pro_path = path / "generated_training_data_PRO_0.jsonl"

    generated_args = []
    negatives = []

    with open(con_path, "r") as f:
        generated_args += ndjson.load(f)

    with open(pro_path, "r") as f:
        generated_args += ndjson.load(f)

    aspects = set(a["aspect"] for a in generated_args)

    for arg in generated_args:
        negative = arg.copy()

        if random.random() < 0.5:
            negative["stance"] = "PRO" if negative["stance"] == "CON" else "CON"
        else:
            negative_aspects = [a for a in aspects if a != negative["aspect"]]
            negative["aspect"] = random.choice(negative_aspects)

        negatives.append(negative)

    return generated_args, negatives


def collect_generated(output, generated, negatives):

    for arg in generated:
        id = str(uuid.uuid4())
        prompt = f"Argument topic: {arg['topic']}.\nStance of the argument: {arg['stance']}.\nTopic aspect: {arg['aspect']}."
        output.append_instance(id, prompt, [arg["sent"]])
        output.append_positive_example(prompt, arg["sent"], "")

    for arg in negatives:
        prompt = f"Generate an argument on topic: {arg['topic']}.\nStance of the argument should be: {arg['stance']}.\nArgument should be related to topic via following aspect: {arg['aspect']}."
        output.append_negative_example(prompt, arg["sent"], "")


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Program to convert gretz20 ibm quality dataset into appropriate form")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    metadata = Metadata(DATASET_NAME)

    dataset_path = (datasets_path() /
                    "ukp-aspect-argument-generation" /
                    "generated_arguments")

    output = Output(DATASET_NAME)
    output.append_definition("Given topic, generate an argument that has appropriate stance. Argument should be related to the topic in a given aspect.")

    common_abortion_path = dataset_path / "common-crawl-en" / "abortion" / "generation_data"
    common_cloning_path = dataset_path / "common-crawl-en" / "cloning" / "generation_data"
    common_death_penalty_path = dataset_path / "common-crawl-en" / "death_penalty" / "generation_data"
    common_gun_control_path = dataset_path / "common-crawl-en" / "gun_control" / "generation_data"
    common_marijuana_legalization_path = dataset_path / "common-crawl-en" / "marijuana_legalization" / "generation_data"
    common_minimum_wage_path = dataset_path / "common-crawl-en" / "minimum_wage" / "generation_data"
    common_nuclear_energy_path = dataset_path / "common-crawl-en" / "nuclear_energy" / "generation_data"
    common_school_uniforms_path = dataset_path / "common-crawl-en" / "school_uniforms" / "generation_data"

    redditcomments_abortion_path = dataset_path / "redditcomments-en" / "abortion" / "generation_data"
    redditcomments_cloning_path = dataset_path / "redditcomments-en" / "cloning" / "generation_data"
    redditcomments_death_penalty_path = dataset_path / "redditcomments-en" / "death_penalty" / "generation_data"
    redditcomments_gun_control_path = dataset_path / "redditcomments-en" / "gun_control" / "generation_data"
    redditcomments_marijuana_legalization_path = dataset_path / "redditcomments-en" / "marijuana_legalization" / "generation_data"
    redditcomments_minimum_wage_path = dataset_path / "redditcomments-en" / "minimum_wage" / "generation_data"
    redditcomments_nuclear_energy_path = dataset_path / "redditcomments-en" / "nuclear_energy" / "generation_data"
    redditcomments_school_uniforms_path = dataset_path / "redditcomments-en" / "school_uniforms" / "generation_data"

    generated, negatives = read_generated(common_abortion_path)
    collect_generated(output, generated, negatives)
    generated, negatives = read_generated(common_cloning_path)
    collect_generated(output, generated, negatives)
    generated, negatives = read_generated(common_death_penalty_path)
    collect_generated(output, generated, negatives)
    generated, negatives = read_generated(common_gun_control_path)
    collect_generated(output, generated, negatives)
    generated, negatives = read_generated(common_marijuana_legalization_path)
    collect_generated(output, generated, negatives)
    generated, negatives = read_generated(common_minimum_wage_path)
    collect_generated(output, generated, negatives)
    generated, negatives = read_generated(common_nuclear_energy_path)
    collect_generated(output, generated, negatives)
    generated, negatives = read_generated(common_school_uniforms_path)
    collect_generated(output, generated, negatives)

    generated, negatives = read_generated(redditcomments_abortion_path)
    collect_generated(output, generated, negatives)
    generated, negatives = read_generated(redditcomments_cloning_path)
    collect_generated(output, generated, negatives)
    generated, negatives = read_generated(redditcomments_death_penalty_path)
    collect_generated(output, generated, negatives)
    generated, negatives = read_generated(redditcomments_gun_control_path)
    collect_generated(output, generated, negatives)
    generated, negatives = read_generated(redditcomments_marijuana_legalization_path)
    collect_generated(output, generated, negatives)
    generated, negatives = read_generated(redditcomments_minimum_wage_path)
    collect_generated(output, generated, negatives)
    generated, negatives = read_generated(redditcomments_nuclear_energy_path)
    collect_generated(output, generated, negatives)
    generated, negatives = read_generated(redditcomments_school_uniforms_path)
    collect_generated(output, generated, negatives)

    output.write_output(DATASET_FILE)
    metadata.add_dataset(DATASET_FILE)
    metadata.add_evaluation_metric("rouge")
    metadata.write_metadata()
