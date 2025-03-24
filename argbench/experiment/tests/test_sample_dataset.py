import unittest
from argbench.experiment.preprocess import *

class TestSampleDataset(unittest.TestCase):

    def testSampleDataset(self):
        test_path = Path("/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/tasks/argument_rating_effectiveness_gaq_ng20/argument_rating_effectiveness_gaq_test_ng20.json")
        df_test_dataset = pd.read_json(test_path, lines=True)
        size_test_dataset = df_test_dataset.shape[0]
        df_test_sample = load_set(test_path, sample_rate=0.1)
        size_sample = df_test_sample.shape[0]
        expected_sample_size = size_test_dataset // 10
        self.assertEqual(expected_sample_size, size_sample)

        df_test_sample = load_set(test_path, sample_size=10)
        self.assertEqual(df_test_sample.shape[0], 10)