#!/usr/bin/env python3
import random
import math
from typing import List, Dict, Tuple

import nltk
import re

from nltk.tokenize.punkt import PunktTrainer, PunktSentenceTokenizer

from argbench.converter.common import Genres, Output, Skills, read_tabular, datasets_path,  Metadata, add_seed_arg, set_seed, \
    split_test_val_train
from argparse import ArgumentParser
from spacy.pipeline import Sentencizer
from spacy.lang.en import English
import stanza


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

    return clause, clause_start, clause_end, argument['conclusion']

def extract_premises(argument, clauses, case_text):
    premises= []

    for clause_id in argument['premises']:
        clause, clause_start, clause_end = extract_clause(clauses, case_text, clause_id)
        premises.append((clause, clause_start, clause_end, clause_id))
    return premises

def clean_text(text):
    return re.sub("(\n|\r|\s)+", " ", text)

def extract_argumentative_clauses(case):
    arguments = case['arguments']
    clauses = case["clauses"]
    case_text = case['text']

    all_argumentative_clauses = []

    for argument in arguments:
        conclusion = extract_conclusions(argument, clauses, case_text)
        all_argumentative_clauses.append(conclusion)
        premises = extract_premises(argument, clauses, case_text)
        all_argumentative_clauses.extend(premises)
    return all_argumentative_clauses

def get_nltk_sentence_segmenter():

    trainer = get_trainer()
    sentence_segmenter = PunktSentenceTokenizer(trainer.get_params())
    def segment_sentences_nlk(case_text : str):
        sentences = sentence_segmenter.span_tokenize(case_text)
        return sentences
    return segment_sentences_nlk

def get_stanza_sentence_segmenter():
    nlp = stanza.Pipeline(lang='en', processors='tokenize')

    def segment_sentences_stanza(case_text: str):
        doc = nlp(case_text)
        sentence_indices = []

        for sent in doc.sentences:

            sentence_indices.append((sent.tokens[0].start_char, sent.tokens[-1].end_char))

        return sentence_indices

    return segment_sentences_stanza

def get_spacy_sentence_segmenter():
    config = {"punct_chars": None}

    nlp = English()
    nlp.add_pipe("sentencizer", config=config)

    def segment_sentences_spacy(case_text: str):
        doc = nlp(case_text)
        sentence_indices = []

        for sent in doc.sents:
            sentence_indices.append((sent.start_char, sent.end_char+1))
        return sentence_indices

    return segment_sentences_spacy

def extract_candidate_argument_units(case:Dict, trainer, segmenter) -> List[Tuple[str, str, str]]:
    case_text = case['text']
    all_argumentative_clauses = extract_argumentative_clauses(case)

    nltk_segmenter = get_nltk_sentence_segmenter()
    spacy_segmenter = get_spacy_sentence_segmenter()
    stanza_segmenter = get_stanza_sentence_segmenter()

    all_candidates = []
    if segmenter == "spacy":
        sentences = spacy_segmenter(case_text)
    elif segmenter == "stanza":
        sentences = stanza_segmenter(case_text)
    else:
        sentences = nltk_segmenter(case_text)

    for sentence_start, sentence_end in sentences:
        sentence = case_text[sentence_start:sentence_end]
        #print(f"{sentence}\n")
        argument_unit_found = False
        for argument_clause, clause_start, clause_end, clause_id in all_argumentative_clauses:
            if sentence_start == clause_start and sentence_end == clause_end:
                print(f"first case: {argument_clause}")
                all_candidates.append(("Argumentative", argument_clause, clause_id))
                argument_unit_found = True
                break
            elif sentence_start <= clause_start < sentence_end:
                if sentence_start < clause_start:
                    prefix = case_text[sentence_start:clause_start]
                    #print(f"second case: prefix {prefix}")
                    all_candidates.append(("Non-argumentative", prefix, None))
                if clause_end <= sentence_end:
                    all_candidates.append(("Argumentative", argument_clause, clause_id))
                    print(f"second case: {argument_clause}")
                    if clause_end < sentence_end:
                        suffix = case_text[clause_end:sentence_end]
                        all_candidates.append(("Non-argumentative", suffix, None))
                        #print(f"second case: suffix{argument_clause}")
                    argument_unit_found = True
                else:
                    print(f"second case: {case_text[clause_start:sentence_end]}")
                    all_candidates.append(("Argumentative", case_text[clause_start:sentence_end], clause_id))
                    argument_unit_found = True
            elif sentence_start < clause_end <= sentence_end:
                argument_clause = case_text[sentence_start:clause_end]
                all_candidates.append(("Argumentative", argument_clause, clause_id))
                argument_unit_found = True
                if clause_end < sentence_end:
                    suffix = case_text[clause_end:sentence_end]
                    all_candidates.append(("Non-argumentative", suffix, None))
                    #print(f"third case: suffix {suffix}")
                print(f"third case: prefix {argument_clause}")
            elif clause_start < sentence_start and clause_end > sentence_end:
                all_candidates.append(("Argumentative", sentence, clause_id))
                argument_unit_found = True
        if not argument_unit_found:
            all_candidates.append(("Non-argumentative", case_text[sentence_start:sentence_end], None))
    return all_candidates

def process_split(DATASET_NAME, dataset, split_name, metadata, trainer, segmenter="nltk"):
    output = Output(DATASET_NAME)

    output.append_definition("""Given the following document, split all of the document into argumentative units and non-argumentative units.
An argumentative unit is a statement that has an argumentative function for example a conclusion or premise.
Prepend each argumentative unit with argumentative: and spans that are not Argumentative with Non-argumentative:.
Output the extracted spans as they are ordered in the given document and separate them by a new line.
Do not add a new formating or enumeration also do not rephrase the argument units. Order the output spans as they appear in the document.""")


    for case_id, case in enumerate(dataset):
        case_text = case['text']
        all_candidates = extract_candidate_argument_units(case, trainer, segmenter)
        case_output = "".join([f"{label}: {clean_text(candidate_clause)}\n" for label, candidate_clause, _ in all_candidates])
        output.append_instance(str(case_id), clean_text(case_text), [case_output])


    output.append_genre(Genres.LEGAL)
    output.append_subarea(Skills.MINING)
    dataset_file = DATASET_FILE_TEMPLATE.format(split=split_name)
    metadata.add_dataset(dataset_file, split_name)
    output.write_output(dataset_file)


def get_trainer():
    trainer = PunktTrainer()
    corpus = """
        Insofar as the applicant complains that the prohibition to meet relatives and other persons amounted to degrading treatment contrary to Article 3 (Art. 3) of the Convention, the Commission finds no separate issue under this provision. 
        It follows that this part of the application is manifestly ill-founded within the meaning of Article 27 para. 2 (Art. 27-2) of the Convention.
        On 9 March 1992 the authorities seised the applicant's passport with reference to Section 7 para. (d) of the Bulgarian Passport Act (for all references to Bulgarian law see below, Relevant domestic law).
        Prosecutor General's request to the National Assembly of 1 July 1992. As to the reasons for imposing detention on remand, it relied on the extent of public exposure of the committed crime, the personality of the performer and the need to secure the applicant's appearance before court, as well as on Sections 50, 177, 180, 196 para. 2 and 207, and Sections 146 to 148 and 152 para. 1 of the Code of Criminal Procedure.
        The hearing took place on 12 January 1995.  The Government were represented by their Agent, Mrs. G. Beleva, and by Mrs. J. Miteva.
        He is represented before the Commission by Mr. L. W. Weh, a lawyer practising in Bregenz.
        4.  I voted for non-violation of Article 8 (art. 8) because I do not see a necessary link between the breach of the requirements of Article 5 para. 1 (art. 5-1) and the interference in the private and family life of Mrs Murray (and her family). I am satisfied with the approach of the Court in regard to Article 8 (art. 8), and, in particular, with its conclusion that the interference was in accordance with the law and that the contested measures pursued a legitimate aim and were necessary in a democratic society (paragraphs 88 to 94 of the judgment).
        """
    trainer.train(corpus, finalize=False, verbose=True)

    abbreviations = "Art., para., Mrs., Mr., 4."
    trainer.train(abbreviations, finalize=False, verbose=True)
    return trainer

if __name__ == "__main__":




    dataset_path = str(datasets_path()
                       / "echr_corpus"
                       / "ECHR_Corpus.json")
    metadata = Metadata(DATASET_NAME)
    with open(dataset_path) as json_file:
        corpus = json.load(json_file)
        test, val, train = split_test_val_train(corpus)
        trainer = get_trainer()
        process_split(DATASET_NAME, test, "test", metadata, trainer, False)
        process_split(DATASET_NAME, train, "train", metadata, trainer, False)
        process_split(DATASET_NAME, val, "val", metadata, trainer, False)


    metadata.add_genre(Genres.LEGAL)
    metadata.add_skill(Skills.MINING)
    metadata.add_evaluation_metric("fscore")
    metadata.write_metadata()
