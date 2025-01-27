#!/usr/bin/env python3
from common import Genres, Output, Subareas, read_tabular, datasets_path, tasks_path, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import uuid

DATASET_NAME = "premise_extraction_ibm_evidence_sentence_eindor20"
DATASET_FILE = "premise_extraction_ibm_evidence_sentence_eindor20.json"

ACCEPTANCE_RATE = 0.7


if __name__ == "__main__":
    argument_parser = ArgumentParser(description="Convert argument mining dataset")
    add_seed_arg(argument_parser)
    args = argument_parser.parse_known_args()[0]
    set_seed(args)

    dataset_path = str(datasets_path()
                    / "ibm-argument-sentences-2"
                    / "ein-dor-20-corpus-wide-argument-mining-a-working-solution.csv")

    dataset = read_tabular(dataset_path)

    output = Output(DATASET_NAME)

    output.append_definition("""Judge if evidence can be used to support or attack the motion.
                              Possible outputs: Accept if evidence can be an argument to support
                              or attack the motion or Reject if the evidence can not be used to attack or support the motion.""")

    metadata = Metadata(DATASET_NAME)

    for row in dataset.iterrows():
        row = row[1]
        prompt = f"Motion: {row['Motion Text']}\nEvidence: {row['Evidence']}"
        acceptance_rate = row["acceptanceRate"]
        id = str(uuid.uuid4())
        if acceptance_rate > ACCEPTANCE_RATE:
            response_prompt = "Accept"
            output.append_positive_example(prompt, response_prompt, "")
        else:
            response_prompt = "Reject"
            output.append_negative_example(prompt, response_prompt, "")

        output.append_instance(id, prompt, [response_prompt])


    metadata.add_dataset(DATASET_FILE)

    metadata.add_evaluation_metric("f1_macro")
    metadata.add_genre(Genres.WIKIPEDIA)
    metadata.add_subarea(Subareas.MINING)

    output.append_genre(Genres.WIKIPEDIA)
    output.append_subarea(Subareas.MINING)
    output.write_output(DATASET_FILE)

    metadata.write_metadata()
