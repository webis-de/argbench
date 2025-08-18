import os
from unittest import *
from argbench.experiment.testing import *
from argbench.experiment.segmentation_metric import *

class TestF1Segment(TestCase):
    def test_f1_score(self):
        document = "If you come to think about it, emails can be count as one of the most benefical results of modern technology"
        Document = """{"output": [{"If you come to think about it,": "Non-argumentative"}, {"emails can be count as one of the most benefical results of modern technology": "Argumentative"}]}"""
        prediction =  {"output":[{"If you come to":"Non-argumentative"},{"think about it, emails can be count as one of the most benefical results of modern technology": "Argumentative"}]}
        metrics = compute_seg_bio_f1_score([prediction], [Document], [document])
        o_recall =  0.5
        o_precision = 1
        i_recall = 14/14
        i_precision = 13/17
        i_f1 =  2* i_recall * i_precision/(i_recall + i_precision)
        print(f"expcted Arg-I {i_f1}")
        o_f1 = 2*o_recall*o_precision/(o_recall + o_precision)
        print(f"expcted Arg-O {o_f1}")
        expected_f1 = (0 + i_f1 + o_f1) /3
        print(metrics)
        self.assertEqual(expected_f1, metrics["fscore"])

    def test_parse(self):
        ground_truth = """Non-argumentative: If you come to think about it,\nArgumentative: emails can be count as one of the most benefical results of modern technology"""
        spans = parse(ground_truth, ["Non-argumentative", "Argumentative"])
        self.assertIn("Argumentative", spans.keys())
        self.assertIn("Non-argumentative", spans.keys())
        self.assertEqual("If you come to think about it,", spans["Non-argumentative"][0])
        self.assertEqual("emails can be count as one of the most benefical results of modern technology", spans["Argumentative"][0])

    def test_match_f1_score(self):
        document = "If you come to think about it, emails can be count as one of the most benefical results of modern technology"
        ground_truth = """Non-argumentative: If you come to think about it,\nArgumentative: emails can be count as one of the most benefical results of modern technology"""
        prediction =  """Non-argumentative: If you come to\nArgumentative: think about it, emails can be count as one of the most benefical results of modern technology"""
        metrics = compute_seg_match_f1_score([prediction], [ground_truth],
                                             ["Non-argumentative", "Argumentative"], ["Non-argumentative"])
        self.assertEqual(0, metrics["fscore"])
        document = "If you come to think about it, emails can be count as one of the most benefical results of modern technology"
        ground_truth = """Non-argumentative: If you come to think about it,\nArgumentative: emails can be count as one of the most benefical results of modern technology"""
        prediction =  """Non-argumentative: If you come to think about it,\nArgumentative: emails can be count as one of the most benefical results of modern technology"""
        metrics = compute_seg_match_f1_score([prediction], [ground_truth],
                                             ["Non-argumentative", "Argumentative"], ["Non-argumentative"])
        self.assertEqual(1, metrics["fscore"])


        document = "If you come to think about it, emails can be count as one of the most benefical results of modern technology"
        ground_truth = """Non-argumentative: If you come to think about it,\nArgumentative: emails can be count as one of the most benefical results of modern technology"""
        prediction =  """Non-argumentative: If you come to think about it,\nArgumentative: emails can be count as one of the most benefical results: Argumentative\n of modern technology"""
        metrics = compute_seg_match_f1_score([prediction], [ground_truth],
                                             ["Non-argumentative", "Argumentative"], ["Non-argumentative"])
        self.assertEqual(0, metrics["fscore"])

        document = "If you come to think about it, emails can be count as one of the most benefical results of modern technology"
        ground_truth = """Non-argumentative: If you come to think about it,\nArgumentative: emails can be count as one of the most benefical results of modern technology"""
        prediction =  """Argumentative: If you come to think about it,\nArgumentative: emails can be count as one of the most benefical results of modern technology"""
        metrics = compute_seg_match_f1_score([prediction], [ground_truth],
                                             ["Non-argumentative", "Argumentative"], ["Non-argumentative"])
        self.assertAlmostEquals(0.67, metrics["fscore"],2)

        document = "If you come to think about it, emails can be count as one of the most benefical results of modern technology"
        ground_truth = """Non-argumentative: If you come to think about it,\nArgumentative: emails can be count as one of the most benefical results of modern technology"""
        prediction =  """Non-argumentative: If you come to think about it,\nArgumentative: emails can be count as one of the most benefical results of modern technology\nArgumentative: I also believe in Science\n"""
        metrics = compute_seg_match_f1_score([prediction], [ground_truth],
                                             ["Non-argumentative", "Argumentative"], ["Non-argumentative"])
        self.assertAlmostEquals(0.67, metrics["fscore"],2)

    def test_real_text(self):

        with open("test_case_1/segmentation_prediction.txt") as file_p:
            prediction = "".join(file_p.readlines())
        with open("test_case_1/segmentation_ground_truth.txt") as file_gt:
            ground_truth = "".join(file_gt.readlines())
        f1 = compute_seg_match_f1_score([prediction], [ground_truth],  ["non-argumentative", "argumentative"], ["non-argumentative"])
        self.assertTrue(f1)

    def test_real_text_web_discourse(self):
        THIS_DIR = os.path.dirname(os.path.abspath(__file__)) + "/test_case_web_discourse"
        with open(THIS_DIR + "/segmentation_prediction.txt") as file_p:
            prediction = "".join(file_p.readlines())
        with open(THIS_DIR + "/segmentation_ground_truth.txt") as file_gt:
            ground_truth = "".join(file_gt.readlines())
        f1 = compute_seg_match_f1_score([prediction], [ground_truth],  ["non-argumentative", "argumentative"], ["non-argumentative"])
        print(f1)
        self.assertAlmostEquals(f1["argumentative-precision"], 2/3)
        self.assertAlmostEquals(f1["argumentative-recall"], 2/5)

class TestF1Sentence(TestCase):
    def test_f1_score_sent(self):
        document = "I always like playing footbal\nGo to the moon babyboy.\nHit the beast"
        prediction = """Anecdote: I always like playing footbal
        Common Ground: Go to the moon babyboy. 
        Common Ground: Hit the beast"""
        ground_truth_argumnet = """Anecdote: I always like playing footbal
        Anecdote: Go to the moon babyboy.
        Common Ground: Hit the beast"""
        metrics = compute_sentence_f1([prediction], [ground_truth_argumnet], [document])
        anecdote_recall =  0.5
        anecdote_precision = 1
        common_ground_recall = 1
        common_ground_precision = 0.5
        a_f1 =  2* anecdote_precision * anecdote_recall/(anecdote_precision + anecdote_recall)
        print(f"expcted anecdote {a_f1}")
        c_f1 = 2*common_ground_recall*common_ground_precision/(common_ground_precision + common_ground_recall)
        print(f"expcted cc {c_f1}")
        expected_f1 = ( a_f1 + c_f1) / 2
        self.assertEqual(expected_f1, metrics["fscore"])

class TestF1Score(TestCase):
    def test_f1_score(self):
        predictions = ["pro", "Con I", "pro"]
        ground_truth= ["pro", "con", "con"]
        f1 = compute_f1_score(predictions, ground_truth)

        pro_precision = 0.5
        con_precision = 1
        pro_recall = 1
        con_recall = 0.5

        con_f1 = 2*con_precision*con_recall/(con_recall + con_precision)
        pro_f1 = 2*pro_precision*pro_recall/(pro_recall + pro_precision)

        expected_f1 = (con_f1 + pro_f1)/2
        returned_f1 = f1["fscore"]
        print(f"returned {returned_f1}")
        print(f"expected {expected_f1}")
        self.assertEqual(returned_f1, expected_f1)
        self.assertTrue(True)