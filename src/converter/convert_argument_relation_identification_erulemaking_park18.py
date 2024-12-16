import json
from common import Output, add_seed_arg, set_seed, datasets_path, Metadata, Genres, Subareas
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
                current_text = proposition.get("text")

                supporting_sentences = proposition.get("reasons")
                if supporting_sentences:
                    for support_id in supporting_sentences:
                        support_id = int(support_id)
                        supporting_text = proposition_texts.get(support_id)
                        if supporting_text:
                            relations.append(f"Reason:\n{current_text}\n{supporting_text}\n")

                evidence_sentences = proposition.get("evidence")
                if evidence_sentences:
                    for evidence_id in evidence_sentences:
                        evidence_id = int(evidence_id)
                        evidence_text = proposition_texts.get(evidence_id)
                        if evidence_text:
                            relations.append(f"Evidence:\n{current_text}\n{evidence_text}\n")
            if relations:
                output.append_instance(
                    comment_id,
                    ' '.join([prop.get("text") for prop in propositions]),  # 拼接所有的句子作为input
                    ["\n".join(relations)]
                )

            print(f"Processed commentID: {comment_id}")


def main():
    arg_parser = ArgumentParser(description="What dataset will be processed?")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    data_path = (datasets_path() /
                 "erulemaking" /
                 "cdcp_type_edge_annot.jsonlist")

    dataset_name = "argument_relation_identification_erulemaking_extract_park18"
    dataset_file = "argument_relation_identification_erulemaking_extract_park18.json"

    output = Output(dataset_name)
    metadata = Metadata(dataset_name)
    output.append_definition(
        """Detect which sentences provide a reason (support) or evidence for another sentence within the comment.
        Output first Reason or Evidence and then output the sentence pair separated by a new line."""
    )

    process_json_file(data_path, output)

    output.append_genre(Genres.DEBATE_PORTALS)
    output.append_subarea(Subareas.REASONING)
    output.write_output(dataset_file)

    metadata.add_dataset(dataset_file)
    metadata.add_evaluation_metric("rouge")

    metadata.add_genre(Genres.DEBATE_PORTALS)
    metadata.add_subarea(Subareas.REASONING)
    metadata.write_metadata()

    print(f"All files processed and saved to {dataset_file}")


if __name__ == "__main__":
    main()
