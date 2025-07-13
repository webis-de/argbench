import unittest

from argbench.converter.convert_argument_unit_segmentation_echr import *

class TestEchrPreprocessing(unittest.TestCase):
    def test_preprocessing_echr(self):
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

            candidate_argument_units = extract_candidate_argument_units(shortest_case)

            self.assertTrue(shortest_case["text"])
#            self.assertIn(candidate_argument_units, ("Argumentative", "The Government submit that the Austrian reservation to Article 5 (Art. 5) of the Convention prevents the Commission from examining the case."))

            print(" ".join(argument_unit[1] for argument_unit in candidate_argument_units))