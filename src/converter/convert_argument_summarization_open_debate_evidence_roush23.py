import os
import pandas as pd
from common import Output, Metadata, add_seed_arg, set_seed, datasets_path, find_topic_size_to_split, Genres
from argparse import ArgumentParser
from datasets import load_dataset
from tqdm import tqdm

from src.converter.common import Subareas

DATASET_NAME = "argument_summarization_open_debate_evidence_roush23"
DATASET_FILE_TEST = "argument_summarization_open_debate_evidence_test_roush23"
DATASET_FILE_TRAIN = "argument_summarization_open_debate_evidence_train_roush23"
def make_output(dataset, metadata, output_file, split):
    output = Output(DATASET_NAME)

    output.append_definition("Given the following argument, generate a short summary.")

    for i, data in dataset.iterrows():
        id = data['id']
        input_text = data['fulltext']
        output_text = data['tag']
        if not input_text or not output_text:
            continue
        output.append_instance(id, input_text, output_text)
    metadata.add_dataset(output_file, split)
    output.append_genre(Genres.DEBATES)
    output.append_subarea(Subareas.GENERATION)
    output.write_output(output_file)


def process_dataset(cache_directory):
    all_data_frames = []
    for i,file in tqdm(enumerate(os.listdir(cache_directory))):
        if file.endswith("csv"):
            df = pd.read_csv(os.path.join(cache_directory,file))
            df = df.sample(5000)
            all_data_frames.append(df)

    df_all = pd.concat(all_data_frames)
    df_test, df_train = find_topic_size_to_split(df_all, "block")


    metadata = Metadata(DATASET_NAME)
    metadata.add_genre(Genres.DEBATES)
    metadata.add_subarea(Subareas.GENERATION)
    make_output(df_test, metadata, DATASET_FILE_TEST, "test")
    make_output(df_train, metadata, DATASET_FILE_TRAIN, "train")
    metadata.write_metadata()

if __name__ == "__main__":
    # Argument parser setup
    arg_parser = ArgumentParser()
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)
    cache_dir = "/bigwork/nhwpajjy/computational-argumentation-tasks-instructions/datasets/openDebateEvidence/datasets--Yusuf5--OpenCaselist/snapshots/751ef23038d6beca927a66c4af5fb8122f2806b5"
    process_dataset(cache_dir)

    print("Finished all tasks")

