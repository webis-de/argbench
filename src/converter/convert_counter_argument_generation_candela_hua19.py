from common import Output, datasets_path, Metadata, add_seed_arg, set_seed
from nltk.tokenize.treebank import TreebankWordDetokenizer
from argparse import ArgumentParser
import ndjson
import pprint

DATASET_NAME = "counter_argument_generation_candela_hua19"

def process_dataset(data_file, output_file, split_name, metadata):
    output = Output(DATASET_NAME)
    output.append_definition("Write a counterargument to original post and take into account retrieved passages related to the post.")
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
        for retrieved_sentence in instance["target_retrieved_passages"][:2]:
            try:
                sentence = tokenizer.detokenize(t for s in retrieved_sentence["sentences"] for t in s)
                retrieved_phrase.append(sentence)
            except Exception:
                pprint.pprint(instance)
                raise Exception()

        retrieved_phrase = "\n".join(retrieved_phrase)

        prompt = f"Original Post: {op}\nRetrieved Passages: {retrieved_phrase}"

        output.append_instance(instance["url"], prompt, [counter_phrase])

    dataset.close()
    output.write_output(output_file)
    metadata.add_evaluation_metric("rouge")
    metadata.add_dataset(output_file, split_name)
    metadata.write_metadata()


if __name__ == "__main__":
    # Input arguments for dataset generation
    arg_parser = ArgumentParser(description="What dataset will be processed?")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args) # Seed random number generation

    data_path = datasets_path() / "candela" # path to data

    metadata = Metadata(DATASET_NAME)

    process_dataset(
        data_path / "train.jsonl",
        "counter_argument_generation_candela_train_hua19.json",
        "train",
        metadata
    )
    print("Train processed")

    process_dataset(
        data_path / "oracle_test.jsonl",
        "counter_argument_generation_candela_test_hua19.json",
        "test",
        metadata
    )
    print("Test processed")

    process_dataset(
        data_path / "dev.jsonl",
        "counter_argument_generation_candela_dev_hua19.json",
        "dev",
        metadata
    )
    print("Dev processed")
