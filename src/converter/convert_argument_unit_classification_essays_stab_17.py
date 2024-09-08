import os
from lxml import etree
from common import Output, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser

def parse_xmi_file(xmi_path):
    """Parse a single XMI file and extract required information."""
    tree = etree.parse(xmi_path)
    root = tree.getroot()

    filename = root.xpath("//argumentation:MetadataAAE/@filename", namespaces=root.nsmap)[0]
    sofa_string = root.xpath("//cas:Sofa/@sofaString", namespaces=root.nsmap)[0]
    units = root.xpath("//argumentation:ArgumentativeDiscourseUnit", namespaces=root.nsmap)

    instance_output = []
    for unit in units:
        unit_type = unit.get("unitType")
        begin = int(unit.get("begin"))
        end = int(unit.get("end"))
        text_segment = sofa_string[begin:end]

        if unit_type == "majorclaim":
            instance_output.append(f"Major Claim: {text_segment}")
        elif unit_type in ["claim-for", "claim-against"]:
            instance_output.append(f"Claim: {text_segment}")
        elif unit_type == "premise":
            instance_output.append(f"Premise: {text_segment}")

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
    metadata.write_metadata()


def main():
    # Input arguments for dataset generation
    arg_parser = ArgumentParser(description="What dataset will be processed?")
    add_seed_arg(arg_parser)
    arg_parser.add_argument("-a", "--custom_argument", help="Your custom argument")
    args = arg_parser.parse_known_args()[0]
    set_seed(args)  # Seed random number generation

    xmi_directory = "/Users/Wangyaxi/Downloads/computational-argumentation-tasks-instructions-main-datasets-essays-argument-mining/datasets/essays-argument-mining"
    dataset_name = "argument_unit_classification_essays_stab_17"
    dataset_file = "argument_unit_classification_essays_stab_17.json"

    output = Output(dataset_name)
    output.append_definition("Given the following essays extract the main claim, claims, and premises. A claim is a controversial statement and the central component of an argument. Premises are reasons for justifying or refuting the claim. A major claim is the central thesis of the essay.")

    # Process XMI files
    process_xmi_files(xmi_directory, output)

    # Write the dataset to a JSON file
    output.write_output(dataset_file)

    # Create and save metadata
    create_metadata(dataset_name, dataset_file)

    print(f"All files processed and saved to {dataset_file}")


if __name__ == "__main__":
    main()
