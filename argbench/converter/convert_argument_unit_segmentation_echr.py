#!/usr/bin/env python3
import random
import math
from typing import List, Dict

import nltk
import re
from argbench.converter.common import Genres, Output, Skills, read_tabular, datasets_path,  Metadata, add_seed_arg, set_seed, \
    split_test_val_train
from argparse import ArgumentParser

import uuid
import json

DATASET_NAME = "argument_unit_segmentation_echr_pouydal_2020"
DATASET_FILE_TEMPLATE = "argument_unit_segmentation_echr_{split}_pouydal_2020.json"


def extract_clause(clauses, case_text, clause_id_to_extract):
    for clause in clauses:
        if clause['_id']==clause_id_to_extract:
            clause_start=clause['start']
            clause_end=clause['end']
            clause=case_text[clause_start:clause_end]
            return clause, clause_start, clause_end

    raise ValueError(f"{clause_id_to_extract} not found!")


def extract_conclusions(argument, clauses, case_text):

    clause, clause_start, clause_end = extract_clause(clauses, case_text, argument['conclusion'])

    return clause, clause_start, clause_end

def extract_premises(argument, clauses, case_text):
    premises= []

    for clause_id in argument['premises']:
        clause, clause_start, clause_end = extract_clause(clauses, case_text, clause_id)
        premises.append((clause, clause_start, clause_end))
    return premises

def clean_text(text):
    return re.sub("(\n|\r|\s)+", " ", text)

def extract_candidate_argument_units(case:Dict) -> List[str]:
    sentence_segmenter = nltk.tokenize.PunktSentenceTokenizer()
    case_text = case['text']
    arguments = case['arguments']
    clauses = case["clauses"]
    all_argumentative_clauses = []
    all_candidates = []
    for argument in arguments:
        conclusion = extract_conclusions(argument, clauses, case_text)
        all_argumentative_clauses.append(conclusion)
        premises = extract_premises(argument, clauses, case_text)
        all_argumentative_clauses.extend(premises)

    sentences = sentence_segmenter.span_tokenize(case_text)

    for sentence_start, sentence_end in sentences:
        argument_unit_found = False
        for argument_clause, clause_start, clause_end in all_argumentative_clauses:
            if sentence_start == clause_start and sentence_end == clause_end:
                print(f"first case: {argument_clause}")
                all_candidates.append(("Argumentative", argument_clause))
                argument_unit_found = True
                break
            elif sentence_start <= clause_start < sentence_end:
                if sentence_start < clause_start:
                    prefix = case_text[sentence_start:clause_start]
                    print(f"second case: prefix {prefix}")
                    all_candidates.append(("Non-argumentative", prefix))
                if clause_end <= sentence_end:
                    all_candidates.append(("Argumentative", argument_clause))
                    print(f"second case: {argument_clause}")
                    if clause_end < sentence_end:
                        suffix = case_text[clause_end:sentence_end]
                        all_candidates.append(("Non-argumentative", suffix))
                        print(f"second case: suffix{argument_clause}")
                    argument_unit_found = True
                else:
                    print(f"second case: {case_text[clause_start:sentence_end]}")
                    all_candidates.append(("Argumentative", case_text[clause_start:sentence_end]))
            elif sentence_start < clause_end <= sentence_end:
                argument_clause = case_text[sentence_start:clause_end]
                all_candidates.append(("Argumentative", argument_clause))
                argument_unit_found = True
                if clause_end < sentence_end:
                    suffix = case_text[clause_end:sentence_end]
                    all_candidates.append(("Non-argumentative", suffix))
                    print(f"third case: suffix {suffix}")
                print(f"third case: prefix {argument_clause}")

        if not argument_unit_found:
            all_candidates.append(("Non-argumentative", case_text[sentence_start:sentence_end]))
    return all_candidates

def process_split(DATASET_NAME, dataset, split_name, metadata):
    output = Output(DATASET_NAME)

    output.append_definition("""Given the following document, split all of the document into argumentative units and non-argumentative units.
An argumentative unit is a statement that has an argumentative function for example a conclusion or premise.
Prepend each argumentative unit with argumentative: and spans that are not Argumentative with Non-argumentative:.
Output the extracted spans as they are ordered in the given document and separate them by a new line.
Do not add a new formating or enumeration also do not rephrase the argument units. Order the output spans as they appear in the document.""")


    for case_id, case in enumerate(dataset):
        case_text = case['text']
        all_candidates = extract_candidate_argument_units(case)
        case_output = "".join([f"{label}: {clean_text(candidate_clause)}\n" for label, candidate_clause in all_candidates])
        output.append_instance(str(case_id), clean_text(case_text), [case_output])


    output.append_genre(Genres.LEGAL)
    output.append_subarea(Skills.MINING)
    dataset_file = DATASET_FILE_TEMPLATE.format(split=split_name)
    metadata.add_dataset(dataset_file, split_name)
    output.write_output(dataset_file)


if __name__ == "__main__":


    dataset_path = str(datasets_path()
                       / "echr_corpus"
                       / "ECHR_Corpus.json")
    metadata = Metadata(DATASET_NAME)
    with open(dataset_path) as json_file:
        corpus = json.load(json_file)
        test, val, train = split_test_val_train(corpus)
        process_split(DATASET_NAME, test, "test", metadata)
        process_split(DATASET_NAME, train, "train", metadata)
        process_split(DATASET_NAME, val, "val", metadata)


    metadata.add_genre(Genres.LEGAL)
    metadata.add_skill(Skills.MINING)
    metadata.add_evaluation_metric("fscore")
    metadata.write_metadata()
