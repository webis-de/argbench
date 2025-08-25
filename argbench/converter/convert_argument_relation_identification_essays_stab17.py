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
    unit_text_dict = {}
    unit_position = []
    units_dict= {}
    for unit in units:
        unit_id = unit.get("{http://www.omg.org/XMI}id")  # xmi:id with the correct namespace
        unit_position.append(unit_id)
        if unit_id:
            try:
                begin = int(unit.get("begin"))
                end = int(unit.get("end"))
                unit_text = sofa_string[begin:end]
                unit_text_dict[unit_id] = unit_text
                units_dict[unit_id] = unit
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
        premise_position = unit_position.index(premise_id)
        conclusion_position = unit_position.index(conclusion_id)
        first_position = min([premise_position, conclusion_position])
        last_position = max([premise_position, conclusion_position])
        window_half_size = 2
        if first_position >= window_half_size:
            first_position-=window_half_size
        else:
            first_position = 0

        if last_position < len(unit_position) - window_half_size:
            last_position+= window_half_size
        else:
            last_position= len(unit_position) - 1
        first_id = unit_position[first_position]
        last_id = unit_position[last_position]
        first_unit_begin = int(units_dict[first_id].get("begin"))
        last_unit_end = int(units_dict[last_id].get("end"))
        window_text = sofa_string[first_unit_begin:last_unit_end+1]

        argument_type = argument.get("argumentType")

        premise_text = unit_text_dict.get(premise_id, f"Unknown premise (ID: {premise_id})")
        conclusion_text = unit_text_dict.get(conclusion_id, f"Unknown conclusion (ID: {conclusion_id})")
        instance_inputs.append(f"Premise: {premise_text}\nConclusion: {conclusion_text}\nDocument: {window_text}")
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
    split_path = datasets_path() / "essays-argument-mining" / "train-test-split.csv"
    xmi_directory = datasets_path() / "essays-argument-mining"
    dataset_name = "argument_relation_identification_essays_stab17"
    template_path = "argument_relation_identification_essays_{split}_stab17.json"

    df_split = pd.read_csv(split_path, sep=";")
    print(df_split.info())

    ids = df_split["ID"].values
    df_train = df_split[df_split["SET"]=="TRAIN"]
    df_test = df_split[df_split["SET"]=="TEST"]
    val_ids = df_train["ID"].sample(len(df_test)).values
    df_split["SET"] = df_split.apply(lambda x: "VAL" if x["ID"] in val_ids else x["SET"], axis=1)
    splits = df_split["SET"]
    split_map = {ids[i]:splits[i] for i in range(len(ids))}
    task_definition = """Given the following premise and conclusion and their context.\n
     Classify whether the premise supports or attacks the conclusion.
         Only output Support or Attack."""
    metadata = Metadata(dataset_name)
    for split in ["train", "test", "val"]:
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
