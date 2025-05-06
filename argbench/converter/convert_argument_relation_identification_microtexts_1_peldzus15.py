import os
from random import sample

from common import Output, read_tabular, datasets_path, tasks_path, Metadata, add_seed_arg, set_seed, Genres, Skills
from argparse import ArgumentParser
from lxml import etree
from collections import Counter
import uuid


def read_file(path):
    file = open(path,encoding='utf-8',errors="ignore")
    return " ".join(file)

def extract_spans(tree):
    spans = {}

    for child in tree:
        if child.tag == "edu":
            spans[child.attrib["id"]]= child.text
    return spans


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


def format_instances(spans, argument_relations, document):

    added_relations = []
    instances = []

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
        text = f"Document: {document}\n"
        if relation == "sup" or relation == "add":
            if (spans[src_id], spans[trgt_id], "Support\n") not in added_relations:
                label = "Support"
                text += "\nSource: " + spans[src_id] + "\n"
                text += "Target: " + spans[trgt_id]
                instances.append((text, label))
                added_relations.append((spans[src_id], spans[trgt_id], "Support\n"))
        elif relation =="reb" or relation =="und":
            if (spans[src_id], spans[trgt_id] , "Attack\n") not in added_relations:
                label = "Attack"
                text += "\nSource: " + spans[src_id] + "\n"
                text += "Target: " + spans[trgt_id]
                instances.append((text, label))
                added_relations.append((spans[src_id], spans[trgt_id], "Attack\n"))
    return instances


DATASET_NAME = "argument_relation_identification_microtexts_1_peldzus15"
DATASET_FILE = "argument_relation_identification_microtexts_1_{split}_peldzus15.json"

def write_split(files, split):
    output = Output(DATASET_NAME)

    output.append_definition("""Given the following essay and the appended source and target argument units that appear in the essay.\n
    Output Support if the source argument unit supports the target argument unit, or output Attack if the source attacks the target.\n""")

    for file_name in files:
        print(file_name)
        if file_name.endswith(".txt"):
            file_path = os.path.join(root,file_name)

            xml_path = os.path.join(root,file_name.replace("txt","xml"))
            print(xml_path)
            tree=etree.parse(xml_path).getroot()
            spans = extract_spans(tree)
            argument_relations = extract_argument_relations(tree)
            print(f"{argument_relations}")


            document = read_file(file_path)
            argument_relations_formatted = format_instances(spans, argument_relations, document)
            id = str(uuid.uuid4())
            for i, argument_relation in enumerate(argument_relations_formatted):
                output.append_instance(f"{id}-{i}", argument_relation[0] , [argument_relation[1]])
    output.append_genre(Genres.ESSAYS)
    output.append_subarea(Skills.MINING)
    output.write_output(DATASET_FILE.format(split=split))

if __name__ == "__main__":
    argument_parser = ArgumentParser()
    add_seed_arg(argument_parser)
    args = argument_parser.parse_known_args()[0]
    set_seed(args)

    dataset_path = datasets_path() / "microtexts-1/original/original/corpus/en"

    metadata = Metadata(DATASET_NAME)
    metadata.add_dataset(DATASET_FILE.format(split="train"), "train")
    metadata.add_dataset(DATASET_FILE.replace("{split}", "test"), "test")
    metadata.add_dataset(DATASET_FILE.replace("{split}", "val"), "val")
    metadata.add_genre(Genres.ESSAYS)
    metadata.add_skill(Skills.MINING)
    metadata.add_evaluation_metric("fscore")
    metadata.write_metadata()

    print(dataset_path)
    for root,dirs,files in os.walk(dataset_path):
        size = len(files)
        test_size = size * 2 // 10
        training_size = size - test_size
        indices = range(size)

        test_indices = sample(indices, test_size)
        train_indices = [index for index in indices if index not in test_indices]

        val_indices = sample(train_indices, test_size)
        train_indices = [index for index in train_indices if index not in val_indices]

        test_set = [files[index] for index in test_indices]
        train_set = [files[index] for index in train_indices]
        val_set = [files[index] for index in val_indices]

        write_split(test_set, "test")
        write_split(train_set, "train")
        write_split(val_set, "val")



