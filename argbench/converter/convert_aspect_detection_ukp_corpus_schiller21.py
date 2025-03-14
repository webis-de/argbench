from common import Output, datasets_path, tasks_path, Metadata, add_seed_arg, set_seed, Genres, Skills
from argparse import ArgumentParser
import ndjson
from ast import literal_eval as make_tuple
dataset_name = "aspect_detection_ukp_corpus_schiller21"

def process_split(dataset_files, output_file, metadata, dataset_split):
    output = Output(dataset_name)
    output.append_definition("""Given the following argument,
     split the argument into spans of text that cover an aspect or not.
     An aspect is a small substring of the argument that characterizes the argument.
     Multiple aspects can be found in an argument.
     Generate a dictionary for each span with the span as key and Aspect or Not-aspect as a value.
     Do not rephrase the spans or modify it. Always process the whole argument. 
     In case there is no aspect, simply output the argument as key and Not-aspect as a value. 
     Output the dictionaries as a list with the order the spans appear in the text. The list should be the value of a dictionary with the key output. 
     Do not explain.""")
    for dataset_file in dataset_files:
        with open(dataset_file, "r") as f:
            dataset = ndjson.load(f)
            for row in dataset:
                id = row["hash"]
                argument = row["sentence"]
                prompt = f"Argument: {argument}"
                spans = []
                aspect_output = {"output": spans}
                aspect_end_index = 0
                for aspect_index_tuple in row["aspect_pos"]:
                    if aspect_index_tuple == "no_Aspect":
                        spans += [{argument:"Not-aspect"}]
                        break
                    print(aspect_index_tuple)
                    aspect_index_tuple = make_tuple(aspect_index_tuple)
                    aspect_index = int(aspect_index_tuple[0])
                    aspect_len = int(aspect_index_tuple[1])
                    if aspect_index > aspect_end_index:
                        spans += [{argument[aspect_end_index:aspect_index]:"Not-aspect"}]
                    spans += [{argument[aspect_index:aspect_index+aspect_len]:"Aspect:"}]
                    aspect_end_index = aspect_index + aspect_len
                if aspect_end_index < len(argument):
                    spans += [{argument[aspect_end_index:]:"Not-aspect"}]
                output.append_instance(id, prompt, [aspect_output])
    output.append_genre(Genres.WEB)
    output.append_subarea(Skills.GENERATION)
    output.write_output(output_file)
    metadata.add_dataset(output_file, dataset_split=dataset_split)


if __name__ == "__main__":
    # Input arguments for dataset generation
    arg_parser = ArgumentParser(description="What dataset will be processed?")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args) # Seed random number generation

    data_path = datasets_path() / "ukp-corpus-argument-generation" / "argument_aspect_detection_v1.0" / "in_topic"

    metadata = Metadata(dataset_name)

    process_split(
        [data_path / "train.jsonl", data_path / "dev.jsonl"],
        "aspect_detection_ukp_corpus_train_schiller21.json",
        metadata,
        "train"
    )
    process_split(
        [data_path / "test.jsonl"],
        "aspect_detection_ukp_corpus_test_schiller21.json",
        metadata,
        "test"
    )

    metadata.add_genre(Genres.WEB)
    metadata.add_skill(Skills.GENERATION)
    metadata.add_evaluation_metric("aspect-bio-fscore")
    metadata.write_metadata()
