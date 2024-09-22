from common import Output, add_seed_arg, set_seed, Genres, Subareas, Metadata, datasets_path
from argparse import ArgumentParser
import json

def process_json_file(json_file_path, output):
    """Process the JSON file and append examples to the output."""
    with open(json_file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            comment_id = data.get("commentID")
            propositions = data.get("propositions", [])

            # 维护内容顺序
            ordered_texts = []
            input_texts = []  # 用于保存所有text文本的顺序

            for proposition in propositions:
                text = proposition.get("text")
                prop_type = proposition.get("type")

                input_texts.append(text)  # 将所有text文本按顺序加入input

                if prop_type == "fact":
                    ordered_texts.append(f"Fact: {text}")
                elif prop_type == "testimony":
                    ordered_texts.append(f"Testimony: {text}")
                elif prop_type == "value":
                    ordered_texts.append(f"Value: {text}")
                elif prop_type == "policy":
                    ordered_texts.append(f"Policy: {text}")
                elif prop_type == "resource":
                    ordered_texts.append(f"Resource: {text}")
                else:
                    print(f"Unknown type {prop_type} in proposition ID {proposition.get('id')}")

            # 将所有文本合并为input，并将格式化文本合并为output
            combined_input = ' '.join(input_texts)  # 将input中的文本拼接成一个字符串
            combined_output = '\n'.join(ordered_texts)  # 按类型拼接的格式化文本

            # 将input和output写入到实例中
            output.append_instance(comment_id, combined_input, [combined_output])

            print(f"Processed commentID: {comment_id}")

def main():
    # Input arguments for dataset generation
    arg_parser = ArgumentParser(description="What dataset will be processed?")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)  # Seed random number generation

    data_path = (datasets_path() /
                 "erulemaking" /
                 "cdcp_type_edge_annot.jsonlist")

    # Set name of the dataset to identify it and files of that dataset
    dataset_name = "argument_unit_classification_erulemaking_classes_park16"
    dataset_file = "argument_unit_classification_erulemaking_classes_park16.json"

    # Class for collecting dataset file data
    output = Output(dataset_name)
    metadata = Metadata(dataset_name)
    output.append_definition(
        "Classify each sentence into facts, testimony, values, policies, or resources. Keep the order of the sentences as provided in the text.")

    # Read JSON file and process
    process_json_file(data_path, output)

    metadata.add_genre(Genres.DEBATE_PORTALS)
    metadata.add_subarea(Subareas.MINING)
    metadata.add_dataset(dataset_file)

    output.append_genre(Genres.DEBATE_PORTALS)
    output.append_subarea(Subareas.MINING)

    # Write processed dataset to disk
    output.write_output(dataset_file)

    print(f"All files processed and saved to {dataset_file}")

if __name__ == "__main__":
    main()
