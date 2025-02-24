from common import Output, add_seed_arg, set_seed, Genres, Skills, Metadata, datasets_path, split_test_train
from argparse import ArgumentParser
import json

def process_data(lines, dataset_name, dataset_file_name, metadata, split):
    """Process the JSON file and append examples to the output."""
    output = Output(dataset_name)
    output.append_definition(
        """Classify each sentence into Fact, Testimony, Value, Policy, Resource,or Others.
         Fact is objective proposition 'expressing or dealing with facts or conditions as perceived without distortion by
personal feelings, prejudices, or interpretations.'
         Testimony: objective proposition about the author’s personal state or experience.
         Policy:  proposition proposing a specific course of action to be taken.
         Resource: reference to a source of objective evidence.
         Others: anything that does not full under these categories.
         Keep the order of the sentences as provided in the text.
         Only output with one of these classes. Do not explain.
         """)
    for line in lines:
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
                ordered_texts.append(f"Other: {text}")

        # 将所有文本合并为input，并将格式化文本合并为output
        combined_input = ' '.join(input_texts)  # 将input中的文本拼接成一个字符串
        combined_output = '\n'.join(ordered_texts)  # 按类型拼接的格式化文本

        # 将input和output写入到实例中
        output.append_instance(comment_id, combined_input, [combined_output])
    output.write_output(dataset_file_name)
    metadata.add_dataset(dataset_file_name, split)
    output.append_genre(Genres.WEB_FORUMS)
    output.append_subarea(Skills.MINING)

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
    dataset_name = "argument_unit_classification_erulemaking_park16"
    dataset_file_template = "argument_unit_classification_erulemaking_{split}_park16.json"
    with open(data_path, 'r') as f:
        test, train = split_test_train(list(f))
    # Class for collecting dataset file data


        metadata = Metadata(dataset_name)

        # Read JSON file and process
        data_file_train = dataset_file_template.format(split="train")
        data_file_test = dataset_file_template.format(split="test")
        process_data(train, dataset_name, data_file_train, metadata, "train")
        process_data(test, dataset_name, data_file_test, metadata, "test")
        metadata.add_evaluation_metric("sentence-fscore")
        metadata.add_genre(Genres.WEB_FORUMS)
        metadata.add_skill(Skills.MINING)


        metadata.write_metadata()
        # Write processed dataset to disk


        print(f"All files processed and saved to {data_file_train}")
        print(f"All files processed and saved to {data_file_train}")

if __name__ == "__main__":
    main()
