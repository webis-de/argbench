import unittest

from argbench.converter.archive.convert_argument_unit_segmentation_echr import get_spacy_sentence_segmenter, \
    get_stanza_sentence_segmenter
from argbench.converter.convert_argument_relation_detection_echr import generate_case_output, get_nltk_sentence_segmenter
from argbench.converter.convert_argument_unit_segmentation_echr import *

import numpy as np
from random import randint
class TestEchrPreprocessing(unittest.TestCase):

    def test_preprocessing_echr_random(self):

        dataset_path = str(datasets_path()
                           / "echr_corpus"
                           / "ECHR_Corpus.json")
        with open(dataset_path) as json_file:
            corpus = json.load(json_file)
            test, val, train = split_test_val_train(corpus)
            all_data = test + val + train

            i = 41
            print(i)
            case = all_data[i]
            chunks = extract_candidate_argument_units(case)
            for chunk in chunks:
                units = chunk["candidates"]
                self.assertTrue( chunk["text"])
                #            self.assertIn(candidate_argument_units, ("Argumentative", "The Government submit that the Austrian reservation to Article 5 (Art. 5) of the Convention prevents the Commission from examining the case."))
                print( chunk["text"])
                print("**output**\n")

                print("\n***\n".join(f"{unit[0]}: {unit[1]}" for unit in units))



    def test_preprocessing_echr_shortest(self):

        dataset_path = str(datasets_path()
                           / "echr_corpus"
                           / "ECHR_Corpus.json")
        with open(dataset_path) as json_file:
            corpus = json.load(json_file)
            test, val, train = split_test_val_train(corpus)
            all_data = test + val + train
            shortest_length = 1000000000
            shortest_case =None
            shortest_case_id = None
            for case_id, case in enumerate(all_data):
                case_text = case["text"]
                if len(case_text) <shortest_length:
                    shortest_length = len(case_text)
                    shortest_case = case
                    shortest_case_id = case_id
            argument_units = extract_argumentative_clauses(shortest_case)
            argument_unit_ids = [argument_unit[3] for argument_unit in argument_units ]
            chunks = extract_candidate_argument_units(shortest_case)
#             unit_ids = [unit[2] for unit in units ]
#             for argument_unit_id in argument_unit_ids:
#                 self.assertIn(argument_unit_id, unit_ids)
#
#             for argument_unit in argument_units:
#                 for unit in units:
# #                    print(f"{argument_unit[3]}, {unit[2]}")
#                     if argument_unit[3] == unit[2]:
#                         #print("**match**\n")
#                         #print(f"unit: {unit[1]}\n")
#                         #print(f"argument unit: {argument_unit[0]}\n")
#                         self.assertIn(unit[1], argument_unit[0])
#
            self.assertTrue(shortest_case["text"])
#            self.assertIn(candidate_argument_units, ("Argumentative", "The Government submit that the Austrian reservation to Article 5 (Art. 5) of the Convention prevents the Commission from examining the case."))
            print(shortest_case["text"])
            print("**output**\n")
            for chunk in chunks:
                units = chunk["candidates"]
                text = chunk["text"]
                print(text)
                print("\n***\n".join(f"{unit[0]}: {unit[1]}" for unit in units))


class TestSentenceSegmentation(unittest.TestCase):
    def test_sentence_segmentation(self):
        # nltk_segmenter = get_nltk_sentence_segmenter()
        # spacy_segmenter = get_spacy_sentence_segmenter()

        dataset_path = str(datasets_path()
                           / "echr_corpus"
                           / "ECHR_Corpus.json")
        with open(dataset_path) as json_file:
            corpus = json.load(json_file)
            test, val, train = split_test_val_train(corpus)
            all_data = test + val + train
            shortest_length = 1000000000
            shortest_case =None
            shortest_case_id = None
            for case_id, case in enumerate(all_data):
                case_text = case["text"]
                if len(case_text) <shortest_length:
                    shortest_length = len(case_text)
                    shortest_case = case
                    shortest_case_id = case_id



            nltk_sentences = nltk_segmenter(case_text)
            stancy_sentences = spacy_segmenter(case_text)
            print("nltk\n")
            print("\n****\n")
            for sentence in nltk_sentences:
                print(case_text[sentence[0]:sentence[1]])
                print("\n")
            print("\n****\n")
            print("spacy\n")
            for sentence in stancy_sentences:
                print(case_text[sentence[0]:sentence[1]])
                print("\n")



class TestArgumentRelation(unittest.TestCase):
    def test_argument_relation_detection(self):
        dataset_path = str(datasets_path()
                           / "echr_corpus"
                           / "ECHR_Corpus.json")
        segmenter = get_stanza_sentence_segmenter()
        with open(dataset_path) as json_file:
            corpus = json.load(json_file)
            test, val, train = split_test_val_train(corpus)
            all_data = test + val + train
            shortest_length = 1000000000
            shortest_case =None
            shortest_case_id = None
            for case_id, case in enumerate(all_data):
                case_text = case["text"]
                if len(case_text) <shortest_length:
                    shortest_length = len(case_text)
                    shortest_case = case
            print(shortest_case["text"])
            all_relations = generate_case_output(shortest_case, segmenter)
            for relation in all_relations:
                au_1_begining, au_1_end = relation[0]
                au_2_beining, au_2_end = relation[1]
                case_text = shortest_case['text']
                doc = f"argument unit 1: {clean_text(case_text[au_1_begining:au_1_end]).strip()}  \nargument unit 2: {clean_text(case_text[au_2_beining:au_2_end]).strip()} \n relation:{relation[2]}"
                print(doc)

