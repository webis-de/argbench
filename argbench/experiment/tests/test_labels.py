import unittest
from argbench.experiment.utils import *

class testLabelsSet(unittest.TestCase):
    def testloading(self):
        labels = load_labels_set()
        print(labels)