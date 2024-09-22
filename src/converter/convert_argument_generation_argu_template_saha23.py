from common import Output, tasks_path, Metadata, add_seed_arg, set_seed, datasets_path, Genres, Subareas
from argparse import ArgumentParser
import pickle
from dataclasses import dataclass

DATASET_NAME = "argument_generation_argu_template_saha23"
DATASET_FILE = "argument_generation_argu_template_saha23.json"

@dataclass
class Prompt:
    id: str
    prompt: str
    output: str


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Program to convert gretz20 ibm quality dataset into appropriate form")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    metadata = Metadata(DATASET_NAME)

    output_path = tasks_path()

    splits_path = datasets_path() / "argu" / "argu_generator_keys.pkl"
    data_path = datasets_path() / "argu" / "argu_generator_data.pkl"

    arguments = []

    with open(data_path, "rb") as f:
        data = pickle.load(f)
        for id in data:
            arg_data = data[id]
            argument_label = arg_data["basn_lbl"]
            argument_stance = arg_data["stance"]
            argument_topic = arg_data["topic"]
            argument_template = arg_data["sent_var"]

            arg_facts = []
            for var in arg_data["var_map"]:
                var_prompt = f"{var}: {arg_data['var_map'][var]}"
                arg_facts.append(var_prompt)

            arg_facts = "; ".join(arg_facts)

            prompt = f"Argument Type: {argument_label}\nTopic: {argument_topic}\nStance: {argument_stance}\nArgument Template: {argument_template}\nFacts: {arg_facts}"
            output = arg_data["clean_sent"]
            id = str(id)
            arguments.append(Prompt(id, prompt, output))

    output = Output(DATASET_NAME)
    output.append_genre(Genres.DEBATE_PORTALS)
    output.append_subarea(Subareas.GENERATION)
    output.append_definition("Given argument type, topic, stance and template generate an argument that fills template variables with facts while taking other data into account.")
    for arg in arguments:
        output.append_instance(arg.id, arg.prompt, [arg.output])

    metadata.add_dataset(DATASET_FILE)
    metadata.add_genre(Genres.DEBATE_PORTALS)
    metadata.add_subarea(Subareas.GENERATION)
    output.write_output(DATASET_FILE)
    metadata.write_metadata()
