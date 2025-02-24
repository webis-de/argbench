import pandas as pd

from common import Output, tasks_path, Metadata, add_seed_arg, set_seed, datasets_path, Genres, Skills, find_topic_size_to_split
from argparse import ArgumentParser
import pickle
from dataclasses import dataclass

DATASET_NAME = "argument_generation_argu_saha23"
DATASET_FILE_TRAIN = "argument_generation_argu_train_saha23.json"
DATASET_FILE_TEST = "argument_generation_argu_test_saha23.json"

@dataclass
class Prompt:
    id: str
    prompt: str
    output: str

def process_split(data, data_path):
    output = Output(DATASET_NAME)
    for _, arg_data in data.iterrows():
        id = arg_data["id"]
        argument_label = arg_data["basn_lbl"]
        argument_stance = arg_data["stance"]
        argument_topic = arg_data["topic"]

        arg_facts = []
        for var in arg_data["var_map"]:
            arg_facts.append(arg_data["var_map"][var])

        arg_facts = "; ".join(arg_facts)

        prompt = f"Argument Type: {argument_label}\nTopic: {argument_topic}\nStance: {argument_stance}\nFacts: {arg_facts}"
        completion = arg_data["clean_sent"]
        id = str(id)
        arguments.append(Prompt(id, prompt, completion))
    output.append_definition("Given the following argument type, topic, stance, and facts, generate an argument that holds that stance on the topic and is based on the facts. Do not explain.")
    for arg in arguments:
        output.append_instance(arg.id, arg.prompt, [arg.output])
    output.append_genre(Genres.DEBATE_PORTALS)
    output.append_subarea(Skills.GENERATION)
    output.write_output(data_path)

if __name__ == "__main__":
    arg_parser = ArgumentParser()
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    metadata = Metadata(DATASET_NAME)

    output_path = tasks_path()

    splits_path = datasets_path() / "argu" / "argu_generator_keys.pkl"
    data_path = datasets_path() / "argu" / "argu_generator_data.pkl"

    arguments = []
    records = pd.read_pickle(data_path)
    ids, records = list(records.keys()), list(records.values())

    df_data = pd.DataFrame.from_records(records)
    df_data["id"] = ids
    print(df_data.info())
    df_test, df_train = find_topic_size_to_split(df_data, "topic")
    process_split(df_test, DATASET_FILE_TEST)
    process_split(df_train, DATASET_FILE_TRAIN)
    print(len(df_test))
    print(len(df_train))
    metadata.add_dataset(DATASET_FILE_TEST, "test")
    metadata.add_dataset(DATASET_FILE_TRAIN, "train")
    metadata.add_genre(Genres.DEBATE_PORTALS)
    metadata.add_skill(Skills.GENERATION)
    metadata.add_evaluation_metric("generation-score")
    metadata.write_metadata()
