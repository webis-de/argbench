#!/usr/bin/env python3
import random
import math
from common import Genres, Output, Subareas, read_tabular, datasets_path, tasks_path, Metadata, add_seed_arg, set_seed, \
    find_topic_size_to_split
from argparse import ArgumentParser
import uuid

DATASET_NAME = "premise_extraction_ibm_evidence_sentence_eindor20"
DATASET_FILE_TRAIN = "premise_extraction_ibm_evidence_sentence_train_eindor20.json"
DATASET_FILE_TEST = "premise_extraction_ibm_evidence_sentence_test_eindor20.json"

ACCEPTANCE_RATE = 0.7



def process_split(dataset, dataset_file):
    output = Output(DATASET_NAME)

    output.append_definition("""Judge if evidence can be used to support or attack the motion.
                              Possible outputs: Accept if evidence can be an argument to support
                              or attack the motion or Reject if the evidence can not be used to attack or support the motion.""")



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
    output.append_genre(Genres.WIKIPEDIA)
    output.append_subarea(Subareas.MINING)
    output.write_output(dataset_file)


if __name__ == "__main__":
    argument_parser = ArgumentParser(description="Convert argument mining dataset")
    add_seed_arg(argument_parser)
    args = argument_parser.parse_known_args()[0]
    set_seed(args)

    dataset_path = str(datasets_path()
                    / "ibm-argument-sentences-2"
                    / "ein-dor-20-corpus-wide-argument-mining-a-working-solution.csv")

    dataset = read_tabular(dataset_path)
    df_test, df_train =    find_topic_size_to_split(dataset, "Dominant Concept")


    print(len(df_test))
    print(len(df_train))
    process_split(df_test, DATASET_FILE_TEST)
    process_split(df_train, DATASET_FILE_TRAIN)

    metadata = Metadata(DATASET_NAME)
    metadata.add_dataset(DATASET_FILE_TEST, "test")
    metadata.add_dataset(DATASET_FILE_TRAIN, "train")
    metadata.add_genre(Genres.WIKIPEDIA)
    metadata.add_subarea(Subareas.MINING)
    metadata.write_metadata()
