from unittest import *
from ..testing import *

class TestF1Segment(TestCase):
    def test_f1_score(self):
        document = "If you come to think about it, emails can be count as one of the most benefical results of modern technology"
        argumnet = "emails can be count as one of the most benefical results of modern technology"
        prediction = "think about it, emails can be count as one of the most benefical results of modern"
        metrics = compute_bio_f1_score([prediction], [argumnet], [document])
        o_recall =  0.5
        o_precision = 0.8
        i_recall = 12/13
        i_precision = 0.75
        i_f1 =  2* i_recall * i_precision/(i_recall + i_precision)
        print(f"expcted Arg-I {i_f1}")
        o_f1 = 2*o_recall*o_precision/(o_recall + o_precision)
        print(f"expcted Arg-O {o_f1}")
        expected_f1 = (0 + i_f1 + o_f1) /3
        self.assertEqual(expected_f1, metrics["fscore"])


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