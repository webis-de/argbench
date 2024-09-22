import os
from lxml import etree
from common import Output, Metadata, add_seed_arg, datasets_path, set_seed, Genres, Subareas
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

    instance_output = []
    for argument in arguments:
        premise_id = argument.get("premises")
        conclusion_id = argument.get("conclusion")
        argument_type = argument.get("argumentType")

        premise_text = unit_dict.get(premise_id, f"Unknown premise (ID: {premise_id})")
        conclusion_text = unit_dict.get(conclusion_id, f"Unknown conclusion (ID: {conclusion_id})")

        if argument_type == "supports":
            instance_output.append(f"Support:\n{premise_text}\n{conclusion_text}\n")
        elif argument_type == "attacks":
            instance_output.append(f"Attack:\n{premise_text}\n{conclusion_text}\n")

    combined_output = "\n".join(instance_output)

    return filename, sofa_string, combined_output

def process_xmi_files(xmi_directory, output):
    """Process all XMI files in the directory."""
    for xmi_file in os.listdir(xmi_directory):
        if xmi_file.endswith('.xmi'):
            xmi_path = os.path.join(xmi_directory, xmi_file)
            filename, sofa_string, combined_output = parse_xmi_file(xmi_path)

            output.append_instance(filename, sofa_string, [combined_output])
            print(f"Processed: {filename}")


def create_metadata(dataset_name, dataset_file):
    """Create and save metadata."""
    metadata = Metadata(dataset_name)
    metadata.add_evaluation_metric("f1_macro")  # Choose appropriate evaluation metric
    metadata.add_dataset(dataset_file)
    metadata.add_genre(Genres.ESSAYS)
    metadata.add_subarea(Subareas.REASONING)
    metadata.write_metadata()


def main():
    # Input arguments for dataset generation
    arg_parser = ArgumentParser(description="What dataset will be processed?")
    add_seed_arg(arg_parser)
    arg_parser.add_argument("-a", "--custom_argument", help="Your custom argument")
    args = arg_parser.parse_known_args()[0]
    set_seed(args)  # Seed random number generation

    xmi_directory = datasets_path() / "essays-argument-mining"
    dataset_name = "argument_relation_identification_essays_stab17"
    dataset_file = "argument_relation_identification_essays_stab17.json"

    output = Output(dataset_name)
    output.append_definition(
        "Given the following essay list of pairs of sentences where the second sentence supports or attacks the first. Mark the pair with Support or Attack.")

    # Process XMI files
    process_xmi_files(xmi_directory, output)

    # Write the dataset to a JSON file
    output.append_genre(Genres.ESSAYS)
    output.append_subarea(Subareas.REASONING)
    output.write_output(dataset_file)

    # Create and save metadata
    create_metadata(dataset_name, dataset_file)

    print(f"All files processed and saved to {dataset_file}")


if __name__ == "__main__":
    main()
