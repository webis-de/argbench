#!/usr/bin/env python3
import json
import re

from collections import OrderedDict
from typing import List, Dict, Tuple
from nltk.tokenize.punkt import PunktTrainer
from argbench.converter.common import Genres, Output, Skills, datasets_path, Metadata, split_test_val_train

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


def extract_candidate_argument_units(case:Dict) -> List[Tuple[str, str, str]]:
    case_text = case['text']
    all_argumentative_clauses = extract_argumentative_clauses(case)
    last_argument_unit_index = 0
    argument_unit_counter = 0

    all_candidates = []
    unique_clauses = set()
    unique_clause_texts = set()
    unique_argument_clauses = []
    for clause in all_argumentative_clauses:
        clause_unique_id = f"{clause[1]}_{clause[2]}"
        if clause_unique_id not in unique_clauses and clause[0].strip().lower() not in unique_clause_texts:
            unique_argument_clauses.append(clause)
            unique_clause_texts.add(clause[0].strip().lower())
            unique_clauses.add(clause_unique_id)


    unique_argument_clauses = sorted(unique_argument_clauses, key=lambda x: x[1])
    current_text = ""
    chunks = []
    for argument_clause, clause_start, clause_end, clause_id in unique_argument_clauses:
        print(f"cluase start {clause_start} {clause_end} {clause_id}")
        print(argument_clause)
        if clause_start > last_argument_unit_index:

            unargumentative_span = case_text[last_argument_unit_index:clause_start]
            current_text += unargumentative_span
            if unargumentative_span.strip():

                argument_unit_counter+= 1
                all_candidates.append(("Non-argumentative", unargumentative_span, argument_unit_counter))

        argument_unit_counter+= 1
        all_candidates.append(("Argumentative", argument_clause, argument_unit_counter ))
        current_text += argument_clause
        last_argument_unit_index = clause_end
        if len(current_text) >4000:
            chunks.append({"text":current_text, "candidates": all_candidates})
            current_text = ""
            all_candidates = []

    if last_argument_unit_index != len(case_text):
        unargumentative_span = case_text[last_argument_unit_index:]
        if current_text:
            current_text += unargumentative_span
            all_candidates.append(("Non-argumentative", unargumentative_span, argument_unit_counter))
        else:
            chunks[-1]["text"] += unargumentative_span
            chunks[-1]["candidates"].append(("Non-argumentative", unargumentative_span, argument_unit_counter))

    if current_text:
        chunks.append({"text":current_text, "candidates": all_candidates})
    return chunks

def process_split(DATASET_NAME, dataset, split_name, metadata):
    output = Output(DATASET_NAME)

    output.append_definition("""Given the following document, split all of the document into argumentative units and non-argumentative units.
An argumentative unit is a statement that has an argumentative function for example a conclusion or premise.
Prepend each argumentative unit with argumentative: and spans that are not Argumentative with Non-argumentative:.
Output the extracted spans as they are ordered in the given document and separate them by a new line.
Do not add a new formating or enumeration also do not rephrase the argument units. Order the output spans as they appear in the document.""")

    counter = 0
    for case in dataset:

        for chunk in extract_candidate_argument_units(case):

            all_candidates = chunk["candidates"]
            text = chunk["text"]

            case_output = "".join([f"{label}: {clean_text(candidate_clause)}\n" for label, candidate_clause, _ in all_candidates])
            output.append_instance(str(counter), clean_text(text), [case_output])
            counter += 1

    output.append_genre(Genres.LEGAL)
    output.append_subarea(Skills.MINING)
    dataset_file = DATASET_FILE_TEMPLATE.format(split=split_name)
    metadata.add_dataset(dataset_file, split_name)
    output.write_output(dataset_file)
    metadata.add_evaluation_metric("argument-fscore")


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
        process_split(DATASET_NAME, test, "test", metadata)
        process_split(DATASET_NAME, train, "train", metadata)
        process_split(DATASET_NAME, val, "val", metadata)


    metadata.add_genre(Genres.LEGAL)
    metadata.add_skill(Skills.MINING)
    metadata.add_evaluation_metric("fscore")
    metadata.write_metadata()
