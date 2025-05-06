import os
from lxml import etree
from common import Output, Metadata, add_seed_arg, set_seed, Genres, Skills, datasets_path
from argparse import ArgumentParser
import pandas as pd
def parse_xmi_file(xmi_path):
    """Parse a single XMI file and extract required information."""
    tree = etree.parse(xmi_path)
    root = tree.getroot()

    filename = root.xpath("//argumentation:MetadataAAE/@filename", namespaces=root.nsmap)[0]
    sofa_string = root.xpath("//cas:Sofa/@sofaString", namespaces=root.nsmap)[0]
    units = root.xpath("//argumentation:ArgumentativeDiscourseUnit", namespaces=root.nsmap)

    sentences_with_labels = []
    for unit in units:
        unit_type = unit.get("unitType")
        begin = int(unit.get("begin"))
        end = int(unit.get("end"))
        text_segment = sofa_string[begin:end]

        if unit_type == "majorclaim":
            sentences_with_labels.append((text_segment, "Major Claim"))
        elif unit_type in ["claim-for", "claim-against"]:
            sentences_with_labels.append((text_segment, "Claim"))
        elif unit_type == "premise":
            sentences_with_labels.append((text_segment, "Premise"))



    return filename, sofa_string, sentences_with_labels


def process_xmi_files(xmi_directory, output, split_map, split):
    """Process all XMI files in the directory."""
    for xmi_file in os.listdir(xmi_directory):
        if xmi_file.endswith('.xmi'):
            file = xmi_file.replace(".xmi", "")
            if split_map[file].lower() == split:
                xmi_path = os.path.join(xmi_directory, xmi_file)
                filename, sofa_string, sentence_with_labels = parse_xmi_file(xmi_path)
                for text_span, label in sentence_with_labels:
                    instance = sofa_string + f"\nSentence: {text_span}"
                    output.append_instance(filename, instance, [label])
                print(f"Processed: {filename}")


def edit_metadata(metadata, dataset_file, split):
    """Create and save metadata."""

      # Choose appropriate evaluation metric
    metadata.add_dataset(dataset_file, split)
    metadata.add_genre(Genres.ESSAYS)
    metadata.add_skill(Skills.MINING)
    metadata.add_evaluation_metric("fscore")


def main():
    # Input arguments for dataset generation
    arg_parser = ArgumentParser(description="What dataset will be processed?")
    add_seed_arg(arg_parser)
    arg_parser.add_argument("-a", "--custom_argument", help="Your custom argument")
    args = arg_parser.parse_known_args()[0]
    set_seed(args)  # Seed random number generation
    split_path = datasets_path() / "essays-argument-mining" / "train-test-split.csv"
    df_split = pd.read_csv(split_path, sep=";")
    print(df_split.info())
    ids = df_split["ID"].values
    df_train = df_split[df_split["SET"]=="TRAIN"]
    df_test = df_split[df_split["SET"]=="TEST"]
    val_ids = df_train["ID"].sample(len(df_test)).values

    df_split["SET"] = df_split.apply(lambda x: "VAL" if x["ID"] in val_ids else x["SET"] ,axis=1)
    splits = df_split["SET"].values
    split_map = {ids[i]:splits[i] for i in range(len(ids))}
    xmi_directory = (datasets_path() / "essays-argument-mining")
    dataset_name = "argument_unit_classification_essays_stab17"
    dataset_file_template = "argument_unit_classification_essays_{split}_stab17.json"
    task_definition = """Given the following essay and sentence, classify the sentence into either main claim, claim, or premise.
     A claim is a controversial statement and the central component of an argument.
     A Premise is a reason for justifying or refuting the claim. 
     A major claim is the central thesis of the essay.
     Only output with one of these classes.
     """
    metadata = Metadata(dataset_name)
    for split in ["test", "train", "val"]:
        output = Output(dataset_name)
        output.append_definition(task_definition)
        dataset_file = dataset_file_template.format(split=split)
        # Process XMI files
        process_xmi_files(xmi_directory, output, split_map, split)

        # Write the dataset to a JSON file
        output.write_output(dataset_file)
        output.append_genre(Genres.ESSAYS)
        output.append_subarea(Skills.MINING)

        # Create and save metadata
        edit_metadata(metadata, dataset_file, split)
    metadata.write_metadata()
    print(f"All files processed and saved to {dataset_file}")


if __name__ == "__main__":
    main()
