import os
from uuid import uuid4
from common import Metadata, Output, read_tabular, datasets_path, tasks_path, add_seed_arg, set_seed
from argparse import ArgumentParser
from lxml import etree

def extract_text(tree):
    for child in tree:
        if child.tag.endswith("Sofa"):
            return child.attrib["sofaString"]

def extract_argument_units(tree, text):
    argument_units = []
    for child in tree:
        if child.tag.endswith("ArgumentativeDiscourseUnit"):
            id = int(child.attrib['{http://www.omg.org/XMI}id'])
            begin_index = int(child.attrib["begin"])
            end_index = int(child.attrib["end"])
            argument_units.append((id,text[begin_index:end_index]))
    return argument_units

def extract_argument_relations(tree, argument_units):
    argument_relations = []
    for child in tree:
        if child.tag.endswith("Argument"):
            premise_id = int(child.attrib["premises"])
            conclusion_id = int(child.attrib["conclusion"])
            type = child.attrib["argumentType"]
            argument_relations.append((premise_id, conclusion_id, type))
    return argument_relations

def format_argument_units(argument_units):
    text = ""
    for i, argument_unit in enumerate(argument_units):
        text += f"[{i}]: {argument_unit[1]} \n"
    return text

def format_argument_relations(argument_relations, argument_unit_ids):
    text = ""
    for argument_relation in argument_relations:
        premise_id = argument_relation[0]
        conclusion_id = argument_relation[1]
        argument_relation = argument_relation[2]
        premise_index = argument_unit_ids.index(premise_id)
        conclusion_index = argument_unit_ids.index(conclusion_id)
        if argument_relation =="supports":
            text += f"[{premise_index}] --> [{conclusion_index}]\n"
        elif argument_relation =="attacks":
            text += f"[{premise_index}] /-> [{conclusion_index}]\n"

    return text


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="What dataset will be processed?")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args) # Seed random number generation
    dataset_name = "argument_unit_relation_identification_essays_stab17"
    dataset_file = "argument_unit_relation_identification_essays_stab17.json"

    dataset_path = datasets_path() / "argument-detection/stab17-parsing-argumentation-structures-in-persuasive-essays"
    output = Output(dataset_name)
    output.append_definition("Given the following document and the given argument units with the given ids " +
                             "mark an argument unit referenced with [0] that supports another argument unit that referenced with [1] with the following " +
                              "[0] --> [1] and an argument unit [0] that attacks another argument [1] with [0] /-> [1] ")
    metadata = Metadata(dataset_name)

    for root,dirs,files in os.walk(dataset_path):
        for file_name in files:
            if file_name.endswith(".xmi"):
                xml_path = os.path.join(root,file_name.replace("txt","xmi"))
                tree=etree.parse(xml_path).getroot()
                text = extract_text(tree)
                argument_units = extract_argument_units(tree, text)
                argument_unit_ids = [argument_unit[0] for argument_unit in argument_units]
                argument_units_text = format_argument_units(argument_units)
                argument_relations = extract_argument_relations(tree, text)
                argument_relations_text = format_argument_relations(argument_relations, argument_unit_ids)
                id = str(uuid4())
                output.append_instance(id, text + argument_units_text, [argument_relations_text])

    output.write_output(dataset_file)
    metadata.add_dataset(dataset_file)
    metadata.add_evaluation_metric("f1_macro")
    metadata.write_metadata()
