from common import Genres, Output, Skills, read_tabular, datasets_path, tasks_path, Metadata, add_seed_arg, set_seed, \
    find_topic_size_to_split
from argparse import ArgumentParser
import random
from random import sample
import uuid
import os

DATASET_NAME = "frame_identification_webis_argument_framing_19_ajjour19"

def process_split(df, data_file):
    output = Output(DATASET_NAME)
    output.append_definition("Identify the frame of the following argument on the given topic form the following candidate frames. The frame is main highlighted the aspect of the topic which resonate with as specific audience. Do not explain.")

    for row in df.iterrows():
        row = row[1]
        response = row["frame"]
        # frames = dataset["frame"].unique().tolist()
        frames = df[(df["topic_id"] == row["topic_id"]) & (df["frame"] != row["frame"])]["frame"].unique().tolist() # If topic has no other frames, pick from random
        if not len(frames):
            frames = df[df["frame"] != row["frame"]]["frame"].unique().tolist()
        if len(frames) < 3:
            available_frames = (df[df["frame"] != row["frame"]]["frame"]
                                .unique()
                                .tolist())
            frames += random.sample(available_frames, 3 - len(frames))
        wrong_candidates = random.sample(frames, 3)
        response_candidates = [response] + wrong_candidates
        random.shuffle(response_candidates)
        frame_string = ";".join(response_candidates)
        prompt = f"Premise: {row['premise']}\nConclusion: {row['conclusion']}\nCandidate frames: " + frame_string

        id = str(uuid.uuid4())
        output.append_instance(id, prompt, [response])

    output.append_genre(Genres.DEBATE_PORTALS)
    output.append_subarea(Skills.PERSPECTIVE_ASSESSMENT)
    print(data_file)
    output.write_output(data_file)


if __name__ == "__main__":

    arg_parser = ArgumentParser(description="Program to convert ajjour unit segmentation dataset into appropriate form")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    dataset_path = str(datasets_path()
                    / "webis-argument-framing-19"
                    / "Webis-argument-framing.csv")
    dataset = read_tabular(dataset_path)


    train_data_file = "frame_identification_webis_argument_framing_19_train_ajjour19.json"
    val_data_file = "frame_identification_webis_argument_framing_19_val_ajjour19.json"
    test_data_file = "frame_identification_webis_argument_framing_19_test_ajjour19.json"

    df_test, df_training = find_topic_size_to_split(dataset, "topic_id", 0.2)
    df_val, df_training = find_topic_size_to_split(df_training, "topic_id", 0.25)

    print(len(df_training))
    print(len(df_test
              ))
    process_split(df_training, train_data_file)
    process_split(df_test, test_data_file)
    process_split(df_val, val_data_file)

    metadata = Metadata(DATASET_NAME)
    metadata.add_dataset(train_data_file, "train")
    metadata.add_dataset(test_data_file, "test")
    metadata.add_dataset(val_data_file, "val")

    metadata.add_genre(Genres.DEBATE_PORTALS)
    metadata.add_skill(Skills.PERSPECTIVE_ASSESSMENT)
    metadata.add_evaluation_metric("fscore")
    metadata.write_metadata()

