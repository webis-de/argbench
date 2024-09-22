from common import Output, read_tabular, datasets_path, tasks_path, Metadata, add_seed_arg, set_seed, Genres, Subareas
from argparse import ArgumentParser
import uuid

if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Program to convert ajjour unit segmentation dataset into appropriate form")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    dataset_path = str(datasets_path()
                    / "webis-argument-framing-19"
                    / "Webis-argument-framing.csv")
    dataset = read_tabular(dataset_path)

    dataset_name = "frame_identification_webis_stance_classification_19_ajjour19"
    dataset_file = "frame_identification_webis_stance_classification_19_ajjour19.json"

    output = Output(dataset_name)

    output.append_definition("Judge if the claim supports the topic or not. Possible responses: Pro if the claim supports the topic or Con if the claim is against the topic.")

    metadata = Metadata(dataset_name)

    for row in dataset.iterrows():
        row = row[1]
        prompt = f"Premise: {row['premise']}\nConclusion: {row['conclusion']}"
        response = row["stance"]
        id = str(uuid.uuid4())
        output.append_positive_example(prompt, response, "")

        wrong_stance = "Pro" if response == "Con" else "Con"
        output.append_negative_example(prompt, wrong_stance, "")

        output.append_instance(id, prompt, [response])

    metadata.add_dataset(dataset_file)

    output.append_genre(Genres.DEBATE_PORTALS)
    output.append_subarea(Subareas.MINING)
    metadata.add_evaluation_metric("f1_macro")

    output.write_output(dataset_file)
    metadata.add_genre(Genres.DEBATE_PORTALS)
    metadata.add_subarea(Subareas.MINING)

    metadata.write_metadata()
