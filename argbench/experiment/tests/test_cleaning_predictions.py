from unittest import TestCase

from argbench.experiment.run import *

class test_response_cleaning(TestCase):
    def test_response_cleaning(self):
        output = "Different Topics</think> </think> Different Topics</think> </think> Different Topics."
        #response = clean_for_classification(output, False)
        #print(f"response {response}")


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


        output = "</think> No Similarity Step-by-step explanation: 1. **Identify the topic of each argument**:    - Argument 1: Focuses on environmental effects of offshore drilling.    - Argument 2: Discusses a historical moratorium"
        output = clean_for_classification(output, ["Some Similarity", "High Similarity", "Low Similarity", "No Similarity"])
        self.assertEquals(output, "no similarity")

        output = "</think>\nNo Similarity Step-by-step explanation: 1. **Identify the topic of each argument**:    - Argument 1: Focuses on environmental effects of offshore drilling.    - Argument 2: Discusses a historical moratorium"
        output = clean_for_classification(output, ["Some Similarity", "High Similarity", "Low Similarity", "No Similarity"])
        self.assertEquals(output, "no similarity")


        output = "output:\noutput:\nnot-ad-hominem </think> output: not-ad-hominem "
        output = clean_for_classification(output, ["not-ad-hominem", "ad-hominem"])
        self.assertEquals(output, "not-ad-hominem")
