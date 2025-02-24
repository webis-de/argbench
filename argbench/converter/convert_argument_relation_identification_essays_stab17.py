import os

import pandas as pd
from lxml import etree
from common import Output, Metadata, add_seed_arg, datasets_path, set_seed, Genres, Skills
from argparse import ArgumentParser



def parse_xmi_file(xmi_path):
    """Parse a single XMI file and extract required information."""
    tree = etree.parse(xmi_path)
    root = tree.getroot()

    # Extracting filename and sofaString
    filename = root.xpath("//argumentation:MetadataAAE/@filename", namespaces=root.nsmap)[0]
    sofa_string = root.xpath("//cas:Sofa/@sofaString", namespaces=root.nsmap)[0]

    # Extracting ArgumentativeDiscourseUnits
    units = root.xpath("//argumentation:ArgumentativeDiscourseUnit", namespaces=root.nsmap)
    unit_dict = {}
    for unit in units:
        unit_id = unit.get("{http://www.omg.org/XMI}id")  # xmi:id with the correct namespace
        if unit_id:
            try:
                begin = int(unit.get("begin"))
                end = int(unit.get("end"))
                unit_text = sofa_string[begin:end]
                unit_dict[unit_id] = unit_text
            except (TypeError, ValueError) as e:
                print(f"Error extracting text for unit_id {unit_id}: {e}")
        else:
            print(f"No xmi:id found for ArgumentativeDiscourseUnit")

    # Extracting Arguments (Supports/Attacks)
    arguments = root.xpath("//argumentation:Argument", namespaces=root.nsmap)
    instance_inputs = []
    instance_outputs = []
    for argument in arguments:
        premise_id = argument.get("premises")
        conclusion_id = argument.get("conclusion")
        argument_type = argument.get("argumentType")

        premise_text = unit_dict.get(premise_id, f"Unknown premise (ID: {premise_id})")
        conclusion_text = unit_dict.get(conclusion_id, f"Unknown conclusion (ID: {conclusion_id})")
        instance_inputs.append(f"Document:{sofa_string}\nPremise:{premise_text}\nConclusion:{conclusion_text}\n")
        if argument_type == "supports":
            instance_outputs.append(f"Support")
        elif argument_type == "attacks":
            instance_outputs.append(f"Attack")



    return filename, sofa_string, instance_inputs, instance_outputs

def process_xmi_files(xmi_directory, output, split_map, split):
    """Process all XMI files in the directory."""
    for xmi_file in os.listdir(xmi_directory):
        if xmi_file.endswith('.xmi'):
            file_name = xmi_file.replace(".xmi","")
            if split_map[file_name].lower() == split:
                xmi_path = os.path.join(xmi_directory, xmi_file)
                filename, sofa_string, instance_inputs, instance_outputs = parse_xmi_file(xmi_path)
                for i, instance_input in enumerate(instance_inputs):
                    output.append_instance(f"{filename}-{i}", instance_input, [instance_outputs[i]])
                print(f"Processed: {filename}")


def edit_metadata(metadata, dataset_file, split):
    """Create and save metadata."""
    metadata.add_dataset(dataset_file, split)
    metadata.add_genre(Genres.ESSAYS)
    metadata.add_skill(Skills.MINING)
    metadata.add_evaluation_metric("fscore")


def main():
    # Input arguments for dataset generation
    arg_parser = ArgumentParser()
    add_seed_arg(arg_parser)
    arg_parser.add_argument("-a", "--custom_argument", help="Your custom argument")
    args = arg_parser.parse_known_args()[0]
    set_seed(args)  # Seed random number generation
    split_path = datasets_path() / "essays-argument-mining" / "train-test-split.csv"
    xmi_directory = datasets_path() / "essays-argument-mining"
    dataset_name = "argument_relation_identification_essays_stab17"
    template_path = "argument_relation_identification_essays_{split}_stab17.json"

    df_split = pd.read_csv(split_path, sep=";")
    print(df_split.info())
    ids = df_split["ID"].values
    splits = df_split["SET"].values
    split_map = {ids[i]:splits[i] for i in range(len(ids))}
    task_definition = """Given the following essay in which the given premise and conclusion appear.\n
     Classify whether the premise supports or attacks the conclusion.
         Only output Support or Attack. Do not Explain."""
    metadata = Metadata(dataset_name)
    for split in ["train", "test"]:
        output = Output(dataset_name)
        output.append_definition(task_definition)

        # Process XMI files
        process_xmi_files(xmi_directory, output, split_map, split)

        # Write the dataset to a JSON file
        output.append_genre(Genres.ESSAYS)
        output.append_subarea(Skills.MINING)
        dataset_path = template_path.format(split=split)
        output.write_output(dataset_path)

        # Create and save metadata

        edit_metadata(metadata, dataset_path, split)
    metadata.write_metadata()
    print(f"All files processed and saved to {dataset_path}")


if __name__ == "__main__":
    main()
