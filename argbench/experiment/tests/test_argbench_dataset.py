from unittest import TestCase
from datasets import load_from_disk
import json
import os

from transformers import AutoTokenizer


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
        cot_argbench_dataset_path = f"{main_path}/argbench-prompting-chain-of-thought-small"
        in_task_argbench_dataset_path = f"{main_path}/argbench-in-task"
        cross_task_argbench_dataset_path = f"{main_path}/argbench-leave-one-task"
        zero_shot_dataset = load_from_disk(zero_shot_shot_argbench_dataset_path)
        four_shot_dataset = load_from_disk(few_shot_shot_argbench_dataset_path)
        cot_dataset = load_from_disk(cot_argbench_dataset_path)
        in_task_dataset = load_from_disk(in_task_argbench_dataset_path)
        cross_task_dataset = load_from_disk(cross_task_argbench_dataset_path)
        sorted(zero_shot_dataset["test_counter_argument_generation_candela_hua19"]["id"])
        sorted(four_shot_dataset["test_counter_argument_generation_candela_hua19"]["id"])
        with open("/home/yamen/tmp/zero-shot-indices", "w") as stream:
            json.dump(sorted(zero_shot_dataset["test_counter_argument_generation_candela_hua19"]["id"]), stream)

        with open("/home/yamen/tmp/few-shot-indices", "w") as stream:
            json.dump(sorted(four_shot_dataset["test_counter_argument_generation_candela_hua19"]["id"]), stream)

        assert sorted(zero_shot_dataset["test_counter_argument_generation_candela_hua19"]["id"]) ==  sorted(four_shot_dataset["test_counter_argument_generation_candela_hua19"]["id"])
        assert sorted(cot_dataset["test_counter_argument_generation_candela_hua19"]["id"]) == sorted(four_shot_dataset["test_counter_argument_generation_candela_hua19"]["id"])
        assert sorted(in_task_dataset["test_counter_argument_generation_cmv_hua18"]["id"]) == sorted(cross_task_dataset["test_counter_argument_generation_cmv_hua18"]["id"])
        assert sorted(in_task_dataset["test_argument_unit_segmentation_webDiscourse_ajjour17"]["id"]) == sorted(cross_task_dataset["test_argument_unit_segmentation_webDiscourse_ajjour17"]["id"])
        assert sorted(zero_shot_dataset["test_argument_similarity_ukp_aspect_reimers19"]["id"]) == sorted(cross_task_dataset["test_argument_similarity_ukp_aspect_reimers19"]["id"])
    def test_dataset_truncation(self):
        main_path = "/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/argbench-dataset"
        in_task_dataset_path = f"{main_path}/argbench-in-task"
        tokenizer = AutoTokenizer.from_pretrained("/bigwork/nhwpajjy/pre-trained-models/DeepSeek-R1-Distill-Qwen-7B")
        in_task_dataset = load_from_disk(in_task_dataset_path)

        for record in  in_task_dataset["test_counter_argument_generation_cmv_hua18"]:

            tokens = tokenizer(record["input"])
            print(len(tokens["input_ids"]))
            self.assertGreaterEqual(804,len(tokens["input_ids"]))


