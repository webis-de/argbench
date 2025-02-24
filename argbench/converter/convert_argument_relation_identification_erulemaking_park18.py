import json
from common import Output, add_seed_arg, set_seed, datasets_path, Metadata, Genres, Skills
from argparse import ArgumentParser
from random import sample

def process_json_file(docs, output):
    """Process the JSON file and append examples to the output."""
    for line in docs:
        data = json.loads(line.strip())
        comment_id = data.get("commentID")
        propositions = data.get("propositions", [])

        proposition_texts = {prop.get("id"): prop.get("text") for prop in propositions}

        instances = []
        doc = ' '.join([prop.get("text") for prop in propositions])
        for proposition in propositions:
            current_text = proposition.get("text")

            supporting_sentences = proposition.get("reasons")
            if supporting_sentences:
                for support_id in supporting_sentences:
                    support_id = int(support_id)
                    supporting_text = proposition_texts.get(support_id)
                    if supporting_text:
                        instances.append((f"Document: {doc}\nSource: {supporting_text}\nTarget: {current_text}\n", "Reason"))

            evidence_sentences = proposition.get("evidence")
            if evidence_sentences:
                for evidence_id in evidence_sentences:
                    evidence_id = int(evidence_id)
                    evidence_text = proposition_texts.get(evidence_id)
                    if evidence_text:
                        instances.append((f"Document: {doc}\nSource: {evidence_text}\nTarget: {current_text}\n", "Evidence"))

        if instances:
            for i, instance in enumerate(instances):
                output.append_instance(f"{comment_id}-{i}", instance[0], [instance[1]] )

        print(f"Processed commentID: {comment_id}")

def create_output(dataset_name):
    output = Output(dataset_name)
    output.append_definition(
        """Given the following document and the appended two argument units that appear in the essay.\n
        Output Reason if the source argument unit is a reason for the target argument unit\n
        or output Evidence if the source argument unit is an evidence for the target argument unit. Do not Explain."""
    )
    output.append_genre(Genres.WEB_FORUMS)
    output.append_subarea(Skills.MINING)
    return output
def main():
    arg_parser = ArgumentParser(description="What dataset will be processed?")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)
    dataset_name = "argument_relation_identification_erulemaking_park18"
    dataset_file_test = "argument_relation_identification_erulemaking_test_park18.json"
    dataset_file_train = "argument_relation_identification_erulemaking_train_park18.json"
    output = create_output(dataset_name)
    metadata = Metadata(dataset_name)
    

    data_path = (datasets_path() /
                 "erulemaking" /
                 "cdcp_type_edge_annot.jsonlist")
    with open(data_path, 'r') as docs:
        docs = list(docs)
        indices = range(len(docs))
        test_size = len(docs) * 2 // 10
        test_indices = sample(indices, test_size)
        train_indices = [i for i in range(len(docs)) if i not in test_indices]
        test_docs = [docs[i] for i in test_indices]
        train_docs = [docs[i] for i in train_indices]
        process_json_file(train_docs, output, )
        output.write_output(dataset_file_train)
        output = create_output(dataset_name)
        process_json_file(test_docs, output )
        output.write_output(dataset_file_test)

    metadata.add_dataset(dataset_file_test, "test")
    metadata.add_dataset(dataset_file_train, "train")
    metadata.add_genre(Genres.WEB_FORUMS)
    metadata.add_skill(Skills.MINING)
    metadata.add_evaluation_metric("fscore")
    metadata.write_metadata()




if __name__ == "__main__":
    main()
