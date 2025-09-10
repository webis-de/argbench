from unittest import TestCase

from argbench.experiment.run import *

class test_response_cleaning(TestCase):
    def test_response_cleaning(self):
        output = "Different Topics</think> </think> Different Topics</think> </think> Different Topics."
        response = clean_prediction(output, False)
        print(f"response {response}")


    def test_label_clearning(self):
        output = "Some Similarity</think> </think> Some Similarity.</think> </think> Some Similarity."
        output = clean_for_classification(output, ["Some Similarity", "High Similarity", "Low Similarity", "No Similarity"])
        self.assertEquals(output, "some similarity")


    def test_label_clearning(self):
        output = " Some Similarity</think> </think> Some Similarity.</think> </think> Some Similarity."
        output = clean_for_classification(output, ["Some Similarity", "High Similarity", "Low Similarity", "No Similarity"])
        self.assertEquals(output, "some similarity")

        output = "Output: No Similarity</think> </think> Some Similarity.</think> </think> Some Similarity."
        output = clean_for_classification(output, ["Some Similarity", "High Similarity", "Low Similarity", "No Similarity"])
        self.assertEquals(output, "no similarity")

        output = "a Output: Low Similarity:"
        output = clean_for_classification(output, ["Some Similarity", "High Similarity", "Low Similarity", "No Similarity"])
        self.assertEquals(output, "low similarity")

        output = "Output Low similarity."
        output = clean_for_classification(output, ["Some Similarity", "High Similarity", "Low Similarity", "No Similarity"])
        self.assertEquals(output, "low similarity")

        output = "Output Output Low similarity."
        output = clean_for_classification(output, ["Some Similarity", "High Similarity", "Low Similarity", "No Similarity"])
        self.assertEquals(output, "low similarity")


        output = "Some Similarity.</think> </think> Some Similarity.</think> </think> Some Similarity."
        output = clean_for_classification(output, ["Some Similarity", "High Similarity", "Low Similarity", "No Similarity"])
        self.assertEquals(output, "some similarity")
