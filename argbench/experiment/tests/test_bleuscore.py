from argbench.experiment.testing  import *
from unittest import TestCase

class BleuScoreTest(TestCase):
    def testBleuScore(self):
        score = compute_bleu_score(['it is a white cat .',
                            'wow , this dog is huge .'], ['it is a white kitten .',
                                                          'wowww , the dog is huge !'])
        self.assertTrue(score)
        print(score)