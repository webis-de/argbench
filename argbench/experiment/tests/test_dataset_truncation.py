import unittest
from pathlib import Path

from absl.testing.absltest import TestCase
from datasets import Split
from transformers import AutoTokenizer

from argbench.experiment.archive.prepare_experiment_old import DatasetSplit
from argbench.experiment.prepare_experiment import load_set, truncate_set


class unitetestDataTruncation(TestCase):

    def test_dataset_truncation(self):
        task_path = Path("/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/tasks")
        dataset = load_set("counter_argument_generation_cmv_hua18",task_path, DatasetSplit.TRAIN, sample_size = 1000)
        tokenizer = AutoTokenizer.from_pretrained("/bigwork/nhwpajjy/pre-trained-models/DeepSeek-R1-Distill-Qwen-7B")
        right_index = 0
        for index, record in dataset.iterrows():
            if len(record["input"]) > 5000:
                right_index = index

        print(dataset["input"].iloc[right_index])
        df = truncate_set(dataset,tokenizer, 768)
        print(df["input"].iloc[right_index])
        tokens = tokenizer(df["input"].iloc[right_index])
        self.assertEqual(769,len(tokens["input_ids"]))