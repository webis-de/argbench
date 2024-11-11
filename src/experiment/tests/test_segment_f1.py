from unittest import *
from ..testing import *

class TestF1Segment(TestCase):
    def test_f1_score(self):
        document = "If you come to think about it, emails can be count as one of the most benefical results of modern technology"
        argumnet = "emails can be count as one of the most benefical results of modern technology"
        prediction = "think about it, emails can be count as one of the most benefical results of modern"
        metrics = compute_segmentation_f1_score([prediction], [argumnet], [document])
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