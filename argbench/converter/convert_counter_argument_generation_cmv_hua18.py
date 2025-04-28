from common import Genres, Output, Skills, datasets_path, read_tabular, tasks_path, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import uuid
from lxml import etree

DATASET_NAME = "counter_argument_generation_cmv_hua18"


def process_dataset(op_path, arg_path, keynote_path, output=None):
    if not output:
        output = Output(DATASET_NAME)
        output.append_definition("""Given a statement and relevant evidence, generate a counterargument that attacks to the original argument and highlights the given key phrases. Do not explain.""")

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

        statement = " ".join([s.text.replace(" .",".").replace(" ,", ",").strip() for s in post_sentences])
        keynote = ", ".join([s.text.replace(" .",".").replace(" ,", ",").strip() for s in keynote_sentences])
        contexts = " ".join([s.text.replace(" .",".").replace(" ,", ",").strip() for s in post_contextes])
        prompt = f"Statement: {statement}\nKeyphrases: {keynote}\nEvidence: {contexts}"
        response= ""
        for arg in argument_sentences:
            response += arg.text.strip()
        output.append_instance(id, prompt, [response.strip()])

    return output


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    metadata = Metadata(DATASET_NAME)

    data_folder_path = (datasets_path()
                        / "cmv-counter-argument-generation")

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
    print(f"train {len(train.instances)}")

    print("Valid")
    val = process_dataset(valid_op_path, valid_arg_path, valid_kw_path)
    print(f"train + valid {len(train.instances)}")

    print("Test")
    test = process_dataset(test_op_path, test_arg_path, test_kw_path)



    train.append_genre(Genres.WEB_FORUMS)
    train.append_subarea(Skills.GENERATION)

    test.append_genre(Genres.WEB_FORUMS)
    test.append_subarea(Skills.GENERATION)

    val.append_genre(Genres.WEB_FORUMS)
    val.append_subarea(Skills.GENERATION)

    train.write_output("counter_argument_generation_cmv_train_hua18.json")
    test.write_output("counter_argument_generation_cmv_test_hua18.json")
    val.write_output("counter_argument_generation_cmv_val_hua18.json")

    metadata.add_dataset("counter_argument_generation_cmv_train_hua18.json", "train")
    metadata.add_dataset("counter_argument_generation_cmv_test_hua18.json", "test")
    metadata.add_dataset("counter_argument_generation_cmv_val_hua18.json", "val")

    metadata.add_genre(Genres.WEB_FORUMS)
    metadata.add_skill(Skills.GENERATION)
    metadata.add_evaluation_metric("generation-score")
    

    metadata.write_metadata()
