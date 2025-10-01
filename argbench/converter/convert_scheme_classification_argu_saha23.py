from common import Genres, Output, datasets_path, tasks_path, Metadata, add_seed_arg, set_seed, Skills, split_val_train
from argparse import ArgumentParser
from pathlib import Path
import pickle
from dataclasses import dataclass

DATASET_NAME = "scheme_classification_argu_saha23"

@dataclass
class Prompt:
    id: str
    prompt: str
    output: str


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Program to convert gretz20 ibm quality dataset into appropriate form")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    

    metadata = Metadata(DATASET_NAME)

    dataset_path = datasets_path() / "argu"
    output_path = tasks_path()

    splits_path = dataset_path / "arg_span_and_scheme_data_keys.pkl"
    data_path = dataset_path / "arg_span_and_scheme_data.pkl"

    arguments = []

    with open(data_path, "rb") as f:
        data = pickle.load(f)
        for id in data:
            id = str(id)
            prompt = f"Argument: {data[id]['text']}"
            output = data[id]["idiom"][0]
            arguments.append(Prompt(id, prompt, output))

    splits_data = open(splits_path, "rb")
    splits = pickle.load(splits_data)

    split_type = "topic-strict"

    model_split_idx = {}
    for split_idx in splits[split_type]:
        for split_model in splits[split_type][split_idx]:
            model_split_idx[split_model] = []
            for id in splits[split_type][split_idx][split_model]:
                model_split_idx[split_model].append(str(id))
    val_ids, train_ids = split_val_train(model_split_idx["train"])
    model_split_idx["train"] = train_ids
    model_split_idx["val"] = val_ids

    for split_model in model_split_idx:
        output = Output(DATASET_NAME)
        output.append_definition("Classify the following argument according to  the following Walton's argument schemes.\n"+
                                 "Means from Goal: is a form of practical reasoning used to justify an action by showing that it is a necessary or effective means to achieve a desired goal.\n"+
                                 "Goal from Means: is a form of practical reasoning that justifies an action (the means) by demonstrating its effectiveness in achieving a desired outcome (the goal).\n"+
                                 "From Consequence: is a form of practical reasoning used to justify or critique an action based on its anticipated results.\n"+
                                 "From Source Knowledge: type of defeasible reasoning used to support a claim by citing a knowledgeable or credible source.\n"+
                                 "From Source Authority: is a type of reasoning that asserts a claim is true because a recognized expert in the relevant field says it.\n"+
                                 "Rule or Principle: a form of practical reasoning that justifies a specific action or decision by appealing to a general, established rule, policy, or principle.\n"+
                                 "Only output one of the following argument schemes: means from goal, goal from means, from consequence, from source knowledge, from source authority, rule or principle, and other.")
        for id in model_split_idx[split_model]:
            for arg in arguments:
                if arg.id == id:
                    output = arg.output
                    if output == "means for goal":
                        output = "means from goal"
                    output.append_instance(arg.id, f"Argument: {arg.prompt}", [output])
                    break
        dataset_file = f"scheme_classification_argu_{split_model}_saha23.json"
        metadata.add_dataset(dataset_file, split_model)
        output.append_genre(Genres.DEBATE_PORTALS)
        output.append_subarea(Skills.REASONING)
        metadata.add_genre(Genres.DEBATE_PORTALS)
        metadata.add_skill(Skills.REASONING)
        metadata.add_evaluation_metric("fscore")
        output.write_output(dataset_file)




    splits_data.close()


    metadata.write_metadata()
