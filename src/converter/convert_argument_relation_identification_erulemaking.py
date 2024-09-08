import json
from common import Output, datasets_path, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser


def process_json_file(json_file_path, output):
    """Process the JSON file and append examples to the output."""
    with open(json_file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            comment_id = data.get("commentID")
            propositions = data.get("propositions", [])

            proposition_texts = {prop.get("id"): prop.get("text") for prop in propositions}

            relations = []

            for proposition in propositions:
                supporting_sentences = proposition.get("reasons")
                current_text = proposition.get("text")

                if supporting_sentences:
                    for support_id in supporting_sentences:
                        support_id = int(support_id)  # Ensure it's an integer
                        supporting_text = proposition_texts.get(support_id)

                        if supporting_text:
                            relations.append(f"{current_text}\n{supporting_text}\n")

            if relations:
                output.append_instance(
                    comment_id,
                    ' '.join([prop.get("text") for prop in propositions]),  # Concatenated text for input
                    relations
                )

            print(f"Processed commentID: {comment_id}")


def main():
    arg_parser = ArgumentParser(description="What dataset will be processed?")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    json_file_path = "/Users/Wangyaxi/Downloads/erulemaking/cdcp_type_edge_annot.jsonlist"

    dataset_name = "argument_relation_identification_erulemaking"
    dataset_file = "argument_relation_identification_erulemaking.json"

    output = Output(dataset_name)
    output.append_definition(
        "Detect which sentences provide reasons (support) for another sentence within the comment."
    )

    process_json_file(json_file_path, output)

    output.write_output(dataset_file)

    print(f"All files processed and saved to {dataset_file}")


if __name__ == "__main__":
    main()
