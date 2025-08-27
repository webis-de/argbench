from unittest import TestCase

from argbench.experiment.run import *

class test_response_cleaning(TestCase):
    def test_response_cleaning(self):
        output = "Different Topics</think> </think> Different Topics</think> </think> Different Topics."
        response = clean_prediction(output, False)
        print(f"response {response}")