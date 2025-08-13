import json
import re

from collections import OrderedDict, defaultdict
from typing import List, Dict, Tuple, Set
from nltk.tokenize.punkt import PunktTrainer, PunktSentenceTokenizer


from common import Genres, Output, Skills, datasets_path, Metadata, split_test_val_train, get_stanza_sentence_segmenter, clean_text

import re



DATASET_NAME = "argument_relation_detection_echr_poudyal20"
DATASET_FILE_TEMPLATE = "argument_relation_detection_echr_{split}_poudyal20.json"

def get_nltk_sentence_segmenter():
    sentence_segmenter = PunktSentenceTokenizer()
    def segment_sentences_nlk(case_text : str):
        sentences = sentence_segmenter.span_tokenize(case_text)
        return sentences
    return segment_sentences_nlk

def overlap(sentence, argument):

    for unit in argument:
        unit_start, unit_end = unit[1], unit[2]
        sentence_start, sentence_end = sentence

        if sentence_start <= unit_start <= sentence_end:
            return True
        elif sentence_start <= unit_end <= sentence_end:
            return True
        elif sentence_start == unit_start and sentence_end == unit_end:
            return True
        elif unit_start < sentence_start and unit_end > sentence_end:
            return True

    return False

def generate_argument_relation_candidates(window: List[Tuple[int,int]], arguments: Set[Tuple[str,int,int,int]]):
    all_window_relation = []
    for i, sentence_a in enumerate(window):
        for j, sentence_b in enumerate(window):
            if j <= i:
                continue
            if sentence_b[0] != sentence_a[0] and sentence_b[1] != sentence_a[1]:
                found = False
                for argument in arguments:
                    if overlap(sentence_a, argument) and overlap(sentence_b, argument):
                        all_window_relation.append((sentence_a, sentence_b, "related"))
                        found = True
                if not found:
                    all_window_relation.append((sentence_a, sentence_b, "not-related"))
    return all_window_relation



def generate_case_output(case, segmenter) -> List:
    doc_half_window = 5
    case_text = case['text']
    arguments = extract_arguments(case)

    sentences = list(segmenter(case_text))
    all_argument_relation_candidates = []
    for i, sentence in enumerate(sentences):
        if i >=2 or i < len(sentence)-2:
            window = sentences[i-2:i+2]
            argument_relations_candidates = generate_argument_relation_candidates(window, arguments)
            if i >= doc_half_window:
                start_doc_window = i - doc_half_window
            else:
                start_doc_window = 0
            start_sentence = sentences[start_doc_window]
            if i < len(sentences) - doc_half_window:
                end_doc_window = i + doc_half_window
            else:
                end_doc_window = len(sentences) - 1
            end_sentence = sentences[end_doc_window]
            doc = case_text[start_sentence[0]:end_sentence[1]]

            for relation_candidate in argument_relations_candidates:
                if relation_candidate not in all_argument_relation_candidates:
                    all_argument_relation_candidates.append({"relation":relation_candidate, "document":doc})
    return all_argument_relation_candidates

def process_split(DATASET_NAME, dataset, split_name, metadata):
    print(split_name)
    output = Output(DATASET_NAME)

    output.append_definition(""" Given the following document and two sentences, your task is to judge whether they are part of the same argument. 
     An argument consists of a conclusion and multiple premises. Your task is to judge whether the two sentences are part of the same argument, where one sentence supports or attacks the other.
       Output related if there is an argumentative relation between the two sentences or not-related if not. Only output related or non-related.  
    """)
    segmenter = get_stanza_sentence_segmenter()
    for case_id, case in enumerate(dataset):
        all_argument_relation_candidates = generate_case_output(case, segmenter)
        for i, relation_record in enumerate(all_argument_relation_candidates):
            argument_relation_candidate = relation_record["relation"]
            document = relation_record["document"]
            au_1_begining, au_1_end = argument_relation_candidate[0]
            au_2_beining, au_2_end = argument_relation_candidate[1]
            case_text = case['text']

            doc = f"Document: {clean_text(document)} \nSentence 1: {clean_text(case_text[au_1_begining:au_1_end]).strip()} \nSentence 2: {clean_text(case_text[au_2_beining:au_2_end]).strip()}"


            output.append_instance(i, doc, [argument_relation_candidate[2]])

    dataset_file = DATASET_FILE_TEMPLATE.format(split=split_name)
    output.append_genre(Genres.LEGAL)
    output.append_subarea(Skills.MINING)
    metadata.add_dataset(dataset_file, split_name)
    output.write_output(dataset_file)



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

    return clause, clause_start, clause_end, argument['conclusion']

def extract_premises(argument, clauses, case_text):
    premises= []

    for clause_id in argument['premises']:
        clause, clause_start, clause_end = extract_clause(clauses, case_text, clause_id)
        premises.append((clause, clause_start, clause_end, clause_id))
    return premises


def extract_arguments(case) -> List[Tuple[str, int,int, int]]:
    arguments = case['arguments']
    clauses = case["clauses"]
    case_text = case['text']

    all_arguments = []

    for argument in arguments:
        argument_units = set()
        conclusion = extract_conclusions(argument, clauses, case_text)
        argument_units.add(conclusion)

        premises = extract_premises(argument, clauses, case_text)
        for premise in premises:
            argument_units.add(premise)
        all_arguments.append(argument_units)
    return all_arguments




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