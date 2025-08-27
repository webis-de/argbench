from unittest import TestCase

from nltk.translate.bleu_score import sentence_bleu

from argbench.experiment.testing import *


class TestBertScore(TestCase):
    def test_bert_score(self):
        prediction = "If you come to think about it, emails can be count as one of the most benefical results of modern technology"
        reference = "In my view, email clients is one of the most used modern invention"
        score = compute_bert_score([prediction, prediction], [reference, reference])
        print(score)