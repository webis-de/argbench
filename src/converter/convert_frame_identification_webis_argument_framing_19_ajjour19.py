from common import Output, read_tabular, datasets_path, tasks_path, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import random
import uuid

if __name__ == "__main__":

    arg_parser = ArgumentParser(description="Program to convert ajjour unit segmentation dataset into appropriate form")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    dataset_path = str(datasets_path()
                    / "argument-frame-identification"
                    / "Webis-argument-framing.csv")
    dataset = read_tabular(dataset_path)

    output = Output("frame_identification_webis_argument_framing_19_ajjour19")

    output.append_definition("Identify the frame of the following argument on the given topic form the following candidate frames. The frame is main highlighted the aspect of the topic which resonate with as specific audience.")

    metadata = Metadata("frame_identification_webis_argument_framing_19_ajjour19")

    for row in dataset.iterrows():
        row = row[1]
        response = row["frame"]
        # frames = dataset["frame"].unique().tolist()
        frames = dataset[(dataset["topic_id"] == row["topic_id"]) & (dataset["frame"] != row["frame"])]["frame"].unique().tolist() # If topic has no other frames, pick from random
        if not len(frames):
            frames = dataset[dataset["frame"] != row["frame"]]["frame"].unique().tolist()
        if len(frames) < 3:
            available_frames = (dataset[dataset["frame"] != row["frame"]]["frame"]
                                .unique()
                                .tolist())
            frames += random.sample(available_frames, 3 - len(frames))
        wrong_candidates = random.sample(frames, 3)
        response_candidates = [response] + wrong_candidates
        random.shuffle(response_candidates)
        frame_string = ";".join(response_candidates)
        prompt = f"Premise: {row['premise']}\nConclusion: {row['conclusion']}\nCandidate frames: " + frame_string

        id = str(uuid.uuid4())
        output.append_positive_example(prompt, response, "")

        wrong_frame = random.choice([f for f in wrong_candidates if f != response])
        output.append_negative_example(prompt, wrong_frame, "")

        output.append_instance(id, prompt, [response])

    metadata.add_dataset("frame_identification_webis_argument_framing_19_ajjour19.json")

    metadata.add_evaluation_metric("f1_macro")

    output.write_output("frame_identification_webis_argument_framing_19_ajjour19.json")

    metadata.write_metadata()
