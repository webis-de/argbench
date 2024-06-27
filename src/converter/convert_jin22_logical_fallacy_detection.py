from common import Output, datasets_path, tasks_path, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import uuid

dataset_name = "jin22_logical_fallacy_detection"

FALLACY_MAPPING = {
    "AdHominem": "Ad Hominem",
    "AppealtoAuthority": "Appeal To Authority",
    "AppealtoEmotion": "Appeal To Emotion",
    "FalseCause": "False Cause",
    "Slipperyslope": "Slippery Slope",
    "Slogans": "Slogans"
}

def collect_fallacies(tokens, labels):
    total_string = ""
    fallacies = ""
    is_fallacy = False

    for token, label in zip(tokens, labels):
        total_string += f" {token}"
        if label == "O":
            is_fallacy = False
        elif label[:2] == "B-":
            fallacies += f"\n{FALLACY_MAPPING[label[2:]]}:"
            is_fallacy = True

        if is_fallacy:
            fallacies += f" {token}"

    return total_string.strip(), fallacies.strip()


def process_dataset(dataset, data_file, dataset_type, metadata):
    output = Output(dataset_name)
    output.append_definition("Given argumentative text, extract argument units that are logical fallacies. " +
                             "Possible logical fallacies:" +
                             "Ad Hominem - an irrelevant attack towards the person or some aspect of the person who is making the argument, instead of addressing the argument or position directly. " +
                             "Appeal To Authority - an appeal is made to some form of ethics, authority, or credibility. "  +
                             "Appeal To Emotion - manipulation of the recipient’s emotions in order to win an argument. " +
                             "False Cause - a statement that jumps to a conclusion implying a causal relationship without supporting evidence. " +
                             "Slippery Slope - an informal fallacy wherein a conclusion is drawn about all or many instances of a phenomenon on the basis of one or a few instances of that phenomenon. is an example of jumping to conclusions. " +
                             "Slogans - brief and striking phrase used to provoke excitement of the audience.")

    for row in dataset:
        total_string, collected_fallacies = collect_fallacies(row[0], row[1])
        prompt = f"Text: {total_string}"
        id = str(uuid.uuid4())
        output.append_instance(id, prompt, [collected_fallacies])

    output.write_output(data_file)
    metadata.add_dataset(data_file, dataset_type)


# Reader of CONLL file
def read_conll_file(conll_path):
    sentences = []
    with open(conll_path, "r", encoding="utf-8") as f:
        words, labels, arg_comp, arg_rel = [], [], [], []
        for line in f:
            line = line.strip()
            if not line:
                sentences.append((words, labels, arg_rel, arg_comp))
                words, labels, arg_comp, arg_rel = [], [], [], []
            else:
                splits = line.split("\t")
                ## We skip the Equivalent relationship due to insignificant num of occurrences
                # if splits[2] != "Equivalent":
                ## We skip the Equivalent relationship due to insignificant num of occurrences
                # if splits[2] != "Equivalent":
                words.append(splits[1])
                arg_rel.append(splits[2])
                arg_comp.append(splits[3])
                labels.append(splits[-1])
    return sentences


def load_data(train_path, test_path, dev_path):

    ########## LOADING DATA ##########

    ## Labels definition

    ## 1. Converting annotation in list of tokens and tags
    train_data = read_conll_file(train_path)
    dev_data = read_conll_file(dev_path)
    test_data = read_conll_file(test_path)

    return train_data, test_data, dev_data


if __name__ == "__main__":
    # Input arguments for dataset generation
    arg_parser = ArgumentParser(description="What dataset will be processed?")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args) # Seed random number generation

    data_path = datasets_path() / "argument-detection" / "jin22-logical-fallacy-detection" # path to data

    train_data, test_data, dev_data = load_data(data_path / "train.conll", data_path / "test.conll", data_path / "dev.conll")

    # Set name of the dataset to identify it and files of that dataset
    dataset_file_train = "jin22_logical_fallacy_detection_train.json"
    dataset_file_test = "jin22_logical_fallacy_detection_test.json"
    dataset_file_dev = "jin22_logical_fallacy_detection_dev.json"

    # Class for collecting dataset file data
    # Dataset name specifies folder where dataset will be written
    metadata = Metadata(dataset_name)

    process_dataset(train_data, dataset_file_train, "train", metadata)
    process_dataset(test_data, dataset_file_test, "test", metadata)
    process_dataset(dev_data, dataset_file_dev, "dev", metadata)

    metadata.add_evaluation_metric("rouge")
    metadata.write_metadata()
