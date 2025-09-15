from unittest import TestCase
from datasets import load_from_disk
import json
import os
class testArgBenchDataset(TestCase):
    def test_dataset_size(self):
        main_path = "/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/argbench-dataset"
        argbench_dataset = f"{main_path}/argbench-prompting-zero-shot-small"
        dataset = load_from_disk(argbench_dataset)
        self.assertEquals(dataset["test_counter_argument_generation_candela_hua19"].shape[0], 1000)

    def test_sampling_same(self):
        main_path = "/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/argbench-dataset"
        zero_shot_shot_argbench_dataset_path = f"{main_path}/argbench-prompting-zero-shot-small"
        few_shot_shot_argbench_dataset_path = f"{main_path}/argbench-prompting-four-shot-small"
        zero_shot_dataset = load_from_disk(zero_shot_shot_argbench_dataset_path)
        four_shot_dataset = load_from_disk(few_shot_shot_argbench_dataset_path)

        sorted(zero_shot_dataset["test_counter_argument_generation_candela_hua19"]["id"])
        sorted(four_shot_dataset["test_counter_argument_generation_candela_hua19"]["id"])
        with open("/home/yamen/tmp/zero-shot-indices", "w") as stream:
            json.dump(sorted(zero_shot_dataset["test_counter_argument_generation_candela_hua19"]["id"]), stream)

        with open("/home/yamen/tmp/few-shot-indices", "w") as stream:
            json.dump(sorted(four_shot_dataset["test_counter_argument_generation_candela_hua19"]["id"]), stream)

        assert sorted(zero_shot_dataset["test_counter_argument_generation_candela_hua19"]["id"]) ==  sorted(four_shot_dataset["test_counter_argument_generation_candela_hua19"]["id"])