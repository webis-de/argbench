import os
from common import Output, read_tabular, datasets_path, tasks_path, Metadata, add_seed_arg, set_seed, Genres, Skills
from argparse import ArgumentParser
from lxml import etree
from collections import Counter
import uuid
from random import sample

def read_file(path):
    file = open(path,encoding='utf-8',errors="ignore")
    return " ".join(file)

def extract_sentences(tree):
    sentence = ""
    sentences = {}
    current_edus = []
    end_sentence_punctuations = (".", "!", "?")
    for child in tree:
        if child.tag == "edu":
            current_edus.append((child.attrib["id"], child.text))
            if sentence:
                sentence = sentence + " " + child.text
            else:
                sentence = sentence + child.text
            if child.text.endswith(end_sentence_punctuations):
                for edu in current_edus:
                    sentences[edu[0]] = sentence
                sentence = ""
                current_edus = []

    return sentences


def extract_argument_relations(tree):
    argument_relations = []
    for child in tree:
        if child.tag == "edge":
            argument_relations.append((child.attrib["id"], child.attrib["src"], child.attrib["trg"], child.attrib["type"]))
    return argument_relations

def format_argument_units(argument_units):
    text = ""
    for i, argument_unit in enumerate(argument_units):
        text += f"[{i}]: {argument_unit[1]} \n"
    return text

def clean_node(node):
    if node.startswith("a"):
        return node.replace("a", "e")
    else:
        return node

def lookup_node(node, argument_relations):
    """
    Find the source node for undercutting and adding relations
    :param node: the node to find the source node for
    :param argument_relations: a list of all relations in the documen
    :return: the source node of the relation
    """
    while not node.startswith("e") and not node.startswith("a"):
        for argument_relation in argument_relations:
            if argument_relation[0] == node:
                node = argument_relation[1]
                break
    return node


def format_argument_relations(sentences, argument_relations):
    text = ""
    added_relations = []

    for argument_relation in argument_relations:
        src_id = argument_relation[1]
        trgt_id = argument_relation[2]
        relation = argument_relation[3]
        if src_id.startswith("c"):
            src_id = lookup_node(src_id, argument_relations)
        if trgt_id.startswith("c"):
            trgt_id = lookup_node(trgt_id, argument_relations)
        src_id = clean_node(src_id)
        trgt_id = clean_node(trgt_id)
        print(src_id)

        #src_index = sentence_ids.index(src_id)
        #trgt_index = sentence_ids.index(trgt_id)
        if relation == "sup" or relation == "add":
            if (sentences[src_id], sentences[trgt_id], "Support\n") not in added_relations:
                text += "Support\n"
                text += sentences[src_id] + "\n"
                text += sentences[trgt_id] + "\n"
                added_relations.append((sentences[src_id],sentences[trgt_id], "Support\n"))

        elif relation =="reb" or relation =="und":
            if (sentences[src_id], sentences[trgt_id] ,"Attack\n") not in added_relations:
                text += "Attack\n"
                text += sentences[src_id] + "\n"
                text += sentences[trgt_id] + "\n"
                added_relations.append((sentences[src_id],sentences[trgt_id] ,"Attack\n"))
    return text

DATASET_NAME = "argument_relation_identification_microtexts_2_skeppstedt18"
DATASET_FILE = "argument_relation_identification_microtexts_2_{split}_skeppstedt18.json"

def write_split(files, split):
    output = Output(DATASET_NAME)
    output.append_definition("""Given the following essay list of pairs of sentences 
    where the second sentence supports or attacks the first.
    Output first Support or Attack and then output the sentence pair separated by a new line.
    Do not Explain.
    """)
    for file_name in files:
        print(file_name)
        if file_name.endswith(".txt"):
            file_path = os.path.join(root,file_name)

            xml_path = os.path.join(root,file_name.replace("txt","xml"))
            print(xml_path)
            tree=etree.parse(xml_path).getroot()
            argument_units = extract_sentences(tree)
            argument_relations = extract_argument_relations(tree)
            print(f"{argument_relations}")

            argument_relations_formatted = format_argument_relations(argument_units, argument_relations)
            document = read_file(file_path)
            id = str(uuid.uuid4())
            output.append_instance(id, document , [argument_relations_formatted])
    output.append_genre(Genres.ESSAYS)
    output.append_subarea(Skills.MINING)
    output.write_output(DATASET_FILE.format(split=split))

if __name__ == "__main__":
    argument_parser = ArgumentParser()
    add_seed_arg(argument_parser)
    args = argument_parser.parse_known_args()[0]
    set_seed(args)

    dataset_path = datasets_path() / "microtexts2/corpus/"


    metadata = Metadata(DATASET_NAME)
    metadata.add_dataset(DATASET_FILE.format(split="train"), "train")
    metadata.add_dataset(DATASET_FILE.replace("{split}", "test"), "test")
    metadata.add_genre(Genres.ESSAYS)
    metadata.add_skill(Skills.MINING)
    metadata.write_metadata()


    print(dataset_path)
    for root,dirs,files in os.walk(dataset_path):
        size = len(files)
        test_size = size * 2 // 10
        training_size = size - test_size
        indices = range(size)
        test_indices = sample(indices, test_size)
        train_indices = [index for index in indices if index not in test_indices]

        test_set = [files[index] for index in test_indices]
        train_set = [files[index] for index in train_indices]
        write_split(test_set, "test")
        write_split(train_set, "train")
