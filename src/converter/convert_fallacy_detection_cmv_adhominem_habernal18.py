from common import Genres, Output, Subareas, datasets_path, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import ndjson

if __name__ == "__main__":
    # Input arguments for dataset generation
    arg_parser = ArgumentParser(description="What dataset will be processed?")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args) # Seed random number generation

    data_path = datasets_path() / "cmv-adhominem"

    dataset_name = "fallacy_detection_cmv_adhominem_habernal18"
    dataset_file = "fallacy_detection_cmv_adhominem_habernal18.json"
    fallacy_file = data_path / "exported-3621-sampled-positive-negative-ah-no-context.json"
    output = Output(dataset_name)
    output.append_definition("Classify if the following argument is an ad-hominem (personal attack) or not. Answer with ad-hominem or not-ad-hominem")

    metadata = Metadata(dataset_name)
    with open(fallacy_file, "r") as df:
        post_data = ndjson.load(df)

    for post in post_data:

        if post["violated_rule"] == 2:
            ad_hominem = "ad-hominem"
        else:
            ad_hominem = "not-ad-hominem"
        argument = post["body"]
        output.append_instance(id=post["id"],input=f"Argument: {argument}", output=[ad_hominem])

    output.append_genre(Genres.WIKIPEDIA)
    output.append_genre(Genres.DEBATE_PORTALS)
    output.append_subarea(Subareas.MINING)
    output.write_output(dataset_file)
    metadata.add_genre(Genres.WIKIPEDIA)
    metadata.add_genre(Genres.DEBATE_PORTALS)
    metadata.add_subarea(Subareas.MINING)
    metadata.add_evaluation_metric("f1_macro")
    metadata.write_metadata()

