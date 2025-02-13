from common import Output, datasets_path, Metadata, add_seed_arg, set_seed, Genres, Skills
from nltk.tokenize.treebank import TreebankWordDetokenizer
from argparse import ArgumentParser
import ndjson
import pprint

DATASET_NAME = "counter_argument_generation_candela_hua19"

def process_dataset(data_files, output_file, split_name, metadata):
    output = Output(DATASET_NAME)
    output.append_definition("Write a counterargument to the following original post and take into account retrieved passages related to the post. Do not explain.")
    for data_file in data_files:
        dataset = open(data_file, "r")
        file_instances = ndjson.load(dataset)

        tokenizer = TreebankWordDetokenizer()

        for instance in file_instances:
            op = tokenizer.detokenize(instance["op"])

            counter_phrase = []
            for counter_sentence in instance["target_counterarg"]:
                try:
                    sentence = tokenizer.detokenize(counter_sentence["tokens"])
                    counter_phrase.append(sentence)
                except Exception:
                    print(counter_sentence)
                    raise Exception()
            counter_phrase = " ".join(counter_phrase)

            retrieved_phrase = []
            for retrieved_sentence in instance["target_retrieved_passages"][:1]:
                try:
                    sentence = tokenizer.detokenize(t for s in retrieved_sentence["sentences"][:14] for t in s)
                    retrieved_phrase.append(sentence)
                except Exception:
                    pprint.pprint(instance)
                    raise Exception()

            retrieved_phrase = "\n".join(retrieved_phrase)

            prompt = f"Original Post: {op}\nRetrieved Passages: {retrieved_phrase}"

            output.append_instance(instance["url"], prompt, [counter_phrase])
        print(data_file)
        print(len(output.instances))
        dataset.close()
    output.append_genre(Genres.WEB_FORUMS)
    output.append_subarea(Skills.GENERATION)
    output.write_output(output_file)
    
    metadata.add_dataset(output_file, split_name)


if __name__ == "__main__":
    # Input arguments for dataset generation
    arg_parser = ArgumentParser()
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args) # Seed random number generation

    data_path = datasets_path() / "candela" # path to data

    metadata = Metadata(DATASET_NAME)

    train_datafiles = [data_path / "train.jsonl", data_path / "dev.jsonl"]
    process_dataset(
        train_datafiles,
        "counter_argument_generation_candela_train_hua19.json",
        "train",
        metadata
    )
    print("Train processed")

    process_dataset(
        [data_path / "oracle_test.jsonl"],
        "counter_argument_generation_candela_test_hua19.json",
        "test",
        metadata
    )
    print("Test processed")


    metadata.add_genre(Genres.WEB_FORUMS)
    metadata.add_skill(Skills.GENERATION)
    metadata.write_metadata()
