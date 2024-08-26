from common import Output, read_tabular, datasets_path, tasks_path, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import uuid

dataset_name = "key_point_matching_argkp_2021_barhaim21"
dataset_file = "key_point_matching_argkp_2021_barhaim21.json"

if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Program to convert ajjour unit segmentation dataset into appropriate form")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    dataset_path = str(datasets_path()
                    / "argkpa-2021"
                    / "ArgKP-2021_dataset.csv")
    dataset = read_tabular(dataset_path)

    output = Output(dataset_name)

    output.append_definition("Judge if keypoint summarizes the argument. Possible responses: Match if argument is summarized by key point and No Match if argument is not summarized by key point.")

    metadata = Metadata(dataset_name)

    for row in dataset.iterrows():
        row = row[1]
        prompt = f"Argument: {row['argument']}\nKey Point: {row['key_point']}"
        response = "Match" if row["label"] == 1 else "No Match"
        wrong_response = "No Match" if row["label"] == 1 else "Match"
        id = str(uuid.uuid4())
        output.append_positive_example(prompt, response, "")
        output.append_negative_example(prompt, wrong_response, "")

        output.append_instance(id, prompt, [response])

    metadata.add_evaluation_metric("f1_macro")

    metadata.add_dataset(dataset_file)

    output.write_output(dataset_file)

    metadata.write_metadata()
