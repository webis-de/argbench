#!/usr/bin/env python3
import random
import math

import nltk
import stanza

from common import Genres, Output, Skills, read_tabular, datasets_path,  Metadata, add_seed_arg, set_seed, \
    split_test_val_train
from argparse import ArgumentParser

import uuid
import json

DATASET_NAME = "conclusion_extraction_echr_pouydal_2020"
DATASET_FILE_TEMPLATE = "conclusion_extraction_echr_{split}_pouydal_2020.json"

def get_stanza_sentence_segmenter():
    nlp = stanza.Pipeline(lang='en', processors='tokenize')

    def segment_sentences_stanza(case_text: str):
        doc = nlp(case_text)
        sentence_indices = []

        for sent in doc.sentences:

            sentence_indices.append((sent.tokens[0].start_char, sent.tokens[-1].end_char))

        return sentence_indices

    return segment_sentences_stanza


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


def process_split(DATASET_NAME, dataset, split_name, metadata):
    output = Output(DATASET_NAME)

    output.append_definition("""Given the following document, Judge if the following sentence is a conclusion or not.
     A conclusion is a controversial statement and the central component of an argument""")
    counter = 0
    conclusion_counts = 0
    sentence_segmenter = get_stanza_sentence_segmenter()
    doc_half_window = 5
    for case_id, case in enumerate(dataset):

        case_text = case['text']
        arguments = case['arguments']
        clauses = case["clauses"]
        all_conclusions = []

        for argument in arguments:
            conclusion = extract_conclusions(argument, clauses, case_text)
            all_conclusions.append(conclusion)

        sentences = sentence_segmenter(case_text)

        for i, (sentence_start, sentence_end) in enumerate(sentences):
            sentence = case_text[sentence_start:sentence_end]
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
            prompt = f"Document: {clean_text(doc)}\nSentence: {clean_text(sentence)}"
            for conclusion, conclusion_start, conclusion_end in all_conclusions:
                if sentence_start <= conclusion_start < sentence_end or sentence_start < conclusion_end <= sentence_end:
                    response = "Conclusion"
                    print(f"conclusion: {conclusion}\n")
                    print(f"found in sentence: {sentence}\n")
                    conclusion_counts = conclusion_counts + 1
                else:
                    response = "No-Conclusion"

            output.append_instance(counter, prompt, [response])
            counter += 1
    print(f"conclusion_count={conclusion_counts}")
    output.append_genre(Genres.WIKIPEDIA)
    output.append_subarea(Skills.MINING)
    dataset_file = DATASET_FILE_TEMPLATE.format(split=split_name)
    metadata.add_dataset(dataset_file, split_name)
    output.write_output(dataset_file)
    metadata.add_evaluation_metric("fscore")

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
