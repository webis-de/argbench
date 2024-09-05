import os
import json
from common import Output, add_seed_arg, set_seed
from argparse import ArgumentParser

LABEL_PRIORITY = ['common ground', 'assumption', 'testimony', 'statistics', 'anecdote']

def classify_sentence(labels):
    """
    Classifies the block based on the presence of labels.
    The label with the highest priority is chosen for the block.
    """
    for priority_label in LABEL_PRIORITY:
        if priority_label in labels:
            return priority_label.capitalize()
    return 'Other'


def process_txt_file(txt_file_path, output, filename):
    """Process the annotated text file and append classified sentences to the output."""
    with open(txt_file_path, 'r') as f:
        lines = f.readlines()

    title = lines[0].strip()
    sentence = ""
    block_labels = set()
    for line in lines[1:]:
        parts = line.strip().split('\t')
        if len(parts) < 3:
            continue

        current_label = parts[1].strip().lower()
        text_part = parts[2].strip()

        if text_part == '.' and current_label == "no-unit":
            if sentence:
                block_label = classify_sentence(block_labels)
                output.append_instance(filename, "", [f"{block_label}: {sentence.strip()}"])
            sentence = ""
            block_labels.clear()
        else:
            sentence += " " + text_part
            if current_label in LABEL_PRIORITY:
                block_labels.add(current_label)

    if sentence:
        block_label = classify_sentence(block_labels)
        output.append_instance(filename, "", [f"{block_label}: {sentence.strip()}"])


def process_all_txt_files_in_directory(directory_path, output_directory):
    """Process all .txt files in the specified directory."""
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            txt_file_path = os.path.join(directory_path, filename)
            dataset_name = os.path.splitext(filename)[0]  # Use the filename (without extension) as id
            dataset_file = os.path.join(output_directory, f"{dataset_name}.json")
            print(f"Processing file: {txt_file_path}")

            output = Output(dataset_name)
            output.append_definition(
                "Classify sentences into common ground, assumption, testimony, statistics, anecdote, or other."
            )
            # output.append_metadata("Title", dataset_name)  # Add Title from the filename

            process_txt_file(txt_file_path, output, dataset_name)
            output.write_output(dataset_file)
            print(f"Processed {filename} and saved to {dataset_file}")


def main():
    arg_parser = ArgumentParser(description="Classify sentences into different types of evidence.")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    txt_directory_path = "/Users/Wangyaxi/Downloads/complete-annotated-final"
    output_directory = "/Users/Wangyaxi/Downloads/annotated_sentences_classification"

    process_all_txt_files_in_directory(txt_directory_path, output_directory)

    print(f"All files processed and saved to {output_directory}")


if __name__ == "__main__":
    main()
