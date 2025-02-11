
import os
import json
from common import Metadata, Output, add_seed_arg, set_seed, Genres, Subareas, datasets_path, tasks_path
from argparse import ArgumentParser
from random import sample
from collections import defaultdict
import re

LABEL_PRIORITY = ['common-ground', 'assumption', 'testimony', 'statistics', 'anecdote']

def classify_sentence(labels):
    """
    Classifies the block based on the presence of labels.
    The label with the highest priority is chosen for the block.
    """
    for priority_label in LABEL_PRIORITY:
        if priority_label in labels:
            return priority_label.capitalize()
    return 'Other'

def remove_labels(output_text):
    """
    Remove the label from the output text, leaving only the content.
    """
    # Use regex to remove the label part
    return re.sub(r'^(common ground|assumption|testimony|statistics|anecdote|other):\s*', '', output_text, flags=re.IGNORECASE).strip()

def process_txt_file(txt_file_path, instances_dict, filename):
    """Process the annotated text file and append classified sentences to the instances dictionary."""
    with open(txt_file_path, 'r') as f:
        lines = f.readlines()

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
                output_text = f"{block_label}: {sentence.strip()}{text_part}"
                if filename not in instances_dict:
                    instances_dict[filename] = []
                instances_dict[filename].append(output_text)
            sentence = ""
            block_labels.clear()
        else:
            sentence += " " + text_part
            if current_label in LABEL_PRIORITY:
                block_labels.add(current_label)

    if sentence:
        block_label = classify_sentence(block_labels)
        output_text = f"{block_label}: {sentence.strip()}"
        if filename not in instances_dict:
            instances_dict[filename] = []
        instances_dict[filename].append(output_text)

def process_all_txt_files_in_directory(directory_path, instances_dict):
    """Process all .txt files in the specified directory and add to the instances dictionary."""
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            txt_file_path = os.path.join(directory_path, filename)
            dataset_name = os.path.splitext(filename)[0]  # Use the filename (without extension) as id
            print(f"Processing file: {txt_file_path}")

            process_txt_file(txt_file_path, instances_dict, dataset_name)

def main():
    arg_parser = ArgumentParser(description=
    """Classify each sentence in the given document into the following argument unit tpyes:
    Common Ground, Assumption, Testimony, Statistics, Anecdote, or Other.,  
     """)
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    txt_directory_path = datasets_path() / "editorials" / "txt" / "txt" / "complete-annotated-final"
    dataset_train = "argument_unit_classification_editorials_train_alkhatib16.json"
    dataset_test = "argument_unit_classification_editorials_test_alkhatib16.json"
    dataset_name = "argument_unit_classification_editorials_alkhatib16"

    instances_dict = defaultdict(list)
    process_all_txt_files_in_directory(txt_directory_path, instances_dict)
    task_definition ="""Classify each sentence in the given document into the following argument unit types:
    Common Ground, Assumption, Testimony, Statistics, Anecdote, or Other.
    Common Ground: is common knowledge, a self-evident fact, an accepted truth, or
similar. 
    Assumption: the unit states an assumption, conclusion, judgment, or opinion of the author, a
general observation, possibly false fact, or similar.
    Testimony: The unit gives evidence by stating or quoting that a proposition was made by some
expert, authority, witness, group, organization, or similar.
    Statistics:  The unit gives evidence by stating or quoting the results or conclusions of quantitative re-
search, studies, empirical data analyses, or similar.
    Anecdote: The unit gives evidence by stating personal experience of the author, an anecdote, a
concrete example, an instance, a specific event, or similar.
    Other: The unit does not or hardly adds to the argumentative discourse or it does not match any of
the above classes.
      Only output with one of these classes. Do not explain.
     """
    train_output = Output(dataset_name)
    test_output = Output(dataset_name)
    metadata = Metadata(dataset_name)

    test_output.append_definition( task_definition)
    train_output.append_definition(task_definition)

    # output.append_metadata("Title", "All Data")  # Optional: Add a general title
    indices = len(instances_dict)
    test_size = 2 * len(instances_dict) // 10

    test_indices = sample(list(instances_dict.keys()), test_size)
    train_indices = [index for index in instances_dict if index not in test_indices]

    for instance_id, outputs in instances_dict.items():
        # Join multiple outputs with \n and add to the output
        combined_output = "\n".join(outputs)
        combined_input_tmp = "\n".join(remove_labels(text) for text in outputs)
        combined_input = combined_input_tmp.replace('\n', ' ')
        if instance_id in test_indices:
            test_output.append_instance(instance_id, combined_input, [combined_output])
        else:
            train_output.append_instance(instance_id, combined_input, [combined_output])

    train_output.append_genre(Genres.NEWS)
    train_output.append_subarea(Subareas.MINING)
    train_output.write_output(dataset_train)

    test_output.append_genre(Genres.NEWS)
    test_output.append_subarea(Subareas.MINING)
    test_output.write_output(dataset_test)
    metadata.add_dataset(dataset_test, "test")
    metadata.add_dataset(dataset_train, "train")
    metadata.add_genre(Genres.NEWS)
    metadata.add_subarea(Subareas.MINING)
    metadata.write_metadata()


if __name__ == "__main__":
    main()
