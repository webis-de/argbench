from uuid import uuid4
from common import Output, datasets_path, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import ndjson


def parse_support(support):
    if not support:
        return []
    total_supports = []
    for support_id in support:
        if "_" in support_id:
            support_start, support_end = support_id.split("_")
            support_start = int(support_start)
            support_end = int(support_end)
            for r_id in range(support_start, support_end + 1):
                total_supports.append(r_id)
        else:
             total_supports.append(int(support_id))
    return total_supports

def preprocess_propositions(propositions):
    proposition_data = {}
    for proposition in propositions:
        proposition_data[proposition["id"]] = proposition
        proposition_data[proposition["id"]]["reasons"] = parse_support(proposition["reasons"])
        proposition_data[proposition["id"]]["evidence"] = parse_support(proposition["evidence"])
        proposition_data[proposition["id"]]["comments"] = []

    labeled_propositons = []
    for prop_1, prop_2 in ((p_1, p_2) for p_1 in proposition_data for p_2 in proposition_data if p_1 != p_2):
        comment_text = ""

        if prop_2 in proposition_data[prop_1]["reasons"]:
            label = "[0] --> [1]"
        elif prop_2 in proposition_data[prop_1]["evidence"]:
            label = "[0] |-> [1]"
        else:
            label = "[0] --- [1]"

        for prop in proposition_data:
            if prop == prop_1:
                comment_text += f" [1] {proposition_data[prop]['text']}"
            if prop == prop_2:
                comment_text += f" [0] {proposition_data[prop]['text']}"
            else:
                comment_text += f" {proposition_data[prop]['text']}"

        comment_text = comment_text.strip()
        labeled_propositons.append((comment_text, label))

    return labeled_propositons


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="What dataset will be processed?")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    data_path = (datasets_path() /
                 "argument-detection" /
                 "park18-a-corpus-of-erulemaking-user-comments-for-measruing-evaluability-of-arguments" /
                 "cdcp_type_edge_annot.jsonlist")
    # Set name of the dataset to identify it and files of that dataset
    dataset_name = "argument_unit_classification_erulemaking_park18"
    dataset_file = "argument_unit_classification_erulemaking_park18.json"

    # Class for collecting dataset file data
    # Dataset name specifies folder where dataset will be written
    output = Output(dataset_name)
    output.append_definition("Given the following document and the given argument units with the given ids, mark an argument unit referenced with [0] that provides a reason or rationale for another argument unit that referenced with [1] with the following [0] --> [1]. " +
        "If the argument unit [0] provides an evidence for the argument unit [1], i.e., it proves whether [1] is true or not, mark it with the following [0] |-> [1] " +
        "If the argment unit [0] is not related to argument unit [1], mark it with [0] --- [1]"
    )

    metadata = Metadata(dataset_name)

    with open(data_path, "r") as f:
        reader = ndjson.reader(f)

        for row in reader:
            propositions = preprocess_propositions(row['propositions'])
            for prop_text, prop_label in propositions:
                id = str(row["commentID"]) + "-" + str(uuid4())
                output.append_instance(id, prop_text, [prop_label])

    metadata.add_dataset(dataset_file)
    metadata.add_evaluation_metric("f1_macro")
    output.write_output(dataset_file)
    metadata.write_metadata()
