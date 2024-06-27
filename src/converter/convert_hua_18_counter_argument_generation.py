from common import Output, datasets_path, read_tabular, tasks_path, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import uuid
from lxml import etree

DATASET_NAME = "hua_18_counter_argument_generation"


def process_dataset(op_path, arg_path, keynote_path):
    output = Output(DATASET_NAME)

    output.append_definition("Given statement and relevant evidence, generate a counterargument that relates to the original argument by a given key phrase.")

    op_file = open(op_path, "r")
    arg_file = open(arg_path, "r")
    keynote_file = open(keynote_path, "r")
    html_parser = etree.HTMLParser()

    for post in op_file:
        post_sentences = etree.fromstring(post, html_parser).xpath("//sent")
        post_contextes = etree.fromstring(post, html_parser).xpath("//sent_ctx")
        arguments = next(arg_file)
        argument_sentences = etree.fromstring(arguments, html_parser).xpath("//sent")
        keynotes = next(keynote_file)
        keynote_sentences = etree.fromstring(keynotes, html_parser).xpath("//sent_cs")
        id = str(uuid.uuid4())

        statement = ". ".join([s.text for s in post_sentences])
        keynote = "; ".join([s.text for s in keynote_sentences])
        contexts = "; ".join([s.text for s in post_contextes])
        prompt = f"Statement: {statement}\nKey Phrase: {keynote}\nEvidence: {contexts}"

        for arg in argument_sentences:
            response = arg.text
            output.append_instance(id, prompt, [response])

    return output


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Program to convert gretz20 ibm quality dataset into appropriate form")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    metadata = Metadata(DATASET_NAME)

    data_folder_path = (datasets_path()
                        / "argument-generation"
                        / "hua-18-neural-argument-generation-argumented-with-externally-retrieved-evidence")

    output_folder = tasks_path()

    train_op_path = data_folder_path / "trainable" / "train_core_sample3.src"
    train_arg_path = data_folder_path / "trainable" / "train_core_sample3_arg.tgt"
    train_kw_path = data_folder_path / "trainable" / "train_core_sample3_kp.tgt"

    valid_op_path = data_folder_path / "trainable" / "valid_core_sample3.src"
    valid_arg_path = data_folder_path / "trainable" / "valid_core_sample3_arg.tgt"
    valid_kw_path = data_folder_path / "trainable" / "valid_core_sample3_kp.tgt"

    test_op_path = data_folder_path / "test" / "with_oracle_evidence" / "test.src"
    test_arg_path = data_folder_path / "test" / "with_oracle_evidence" / "test_arg.tgt"
    test_kw_path = data_folder_path / "test" / "with_oracle_evidence" / "test_kp.tgt"

    print("Train")
    train = process_dataset(train_op_path, train_arg_path, train_kw_path)
    print("Valid")
    valid = process_dataset(valid_op_path, valid_arg_path, valid_kw_path)
    print("Test")
    test = process_dataset(test_op_path, test_arg_path, test_kw_path)

    train.write_output("hua_18_counter_argument_generation_train.json")
    valid.write_output("hua_18_counter_argument_generation_valid.json")
    test.write_output("hua_18_counter_argument_generation_test.json")

    metadata.add_dataset("hua_18_counter_argument_generation_train.json", "train")
    metadata.add_dataset("hua_18_counter_argument_generation_test.json", "test")
    metadata.add_dataset("hua_18_counter_argument_generation_dev.json", "dev")

    metadata.add_evaluation_metric("rouge")

    metadata.write_metadata()
