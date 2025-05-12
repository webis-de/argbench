from unittest import *

from transformers import AutoTokenizer

from argbench.experiment.prepare_experiment_old import *
from argbench.experiment.prepare_experiment import *

from argbench.experiment.run import formate_model_template, get_tokenizer


class ExperimentSetupTest(TestCase):

    def test_dataset_load(self):
        df_test,_ = load_set("stance_classification_ibmsc_barhaim17", Path("/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/tasks"), DatasetSplit.TEST)
        df_train,_ = load_set("stance_classification_ibmsc_barhaim17", Path("/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/tasks"), DatasetSplit.TRAIN)
        df_val,_ = load_set("stance_classification_ibmsc_barhaim17", Path("/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/tasks"), DatasetSplit.VAL)
        df_train_val,_ = load_set("stance_classification_ibmsc_barhaim17", Path("/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/tasks"), DatasetSplit.TRAIN_AND_VAL)
        self.assertEqual(len(df_train) + len(df_val), len(df_train_val))


    def test_experiment_setup_in_task_hpo(self):


        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True



        config = RunConfig.from_file(["/bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/experiment/configs/test/barhaim17_mistral_hpo.json"], None)
        train_datasets, test_datasets = collect_datasets(config)
        self.assertEqual(len(test_datasets["stance_classification_ibmsc_barhaim17"]), 262)
        self.assertEqual(list(test_datasets.keys()), ["stance_classification_ibmsc_barhaim17"])
        self.assertEqual(len(train_datasets["stance_classification_ibmsc_barhaim17"]), 435 - 262 + 604)


    def test_experiment_setup_in_task(self):


        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True



        config = RunConfig.from_file(["/bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/experiment/configs/test/barhaim17_mistral.json"], None)
        train_datasets, test_datasets = collect_datasets(config)
        self.assertEqual(len(test_datasets["stance_classification_ibmsc_barhaim17"]), 1355)
        self.assertEqual(list(test_datasets.keys()), ["stance_classification_ibmsc_barhaim17"])
        self.assertEqual(len(train_datasets["stance_classification_ibmsc_barhaim17"]), 435  + 604)

    def test_experiment_setup_corss_task_hpo(self):


        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True



        config = RunConfig.from_file(["/bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/experiment/configs/test/barhaim17_mistral_hpo_cs.json"], None)
        train_datasets, test_datasets = collect_datasets(config)
        self.assertEqual(len(test_datasets["stance_classification_ukp_sentential_stab18"]), 2042)
        self.assertEqual(list(test_datasets.keys()), [
            "warrant_generation_art_chakarbarty21",
            "argument_rating_dagstuhl_15512_overall_quality_wachsmuth17",
            "argument_relation_identification_microtexts_2_skeppstedt18",
            "stance_classification_ukp_sentential_stab18",
            "fallacy_detection_logic_jin22"
        ])


    def test_experiment_setup_corss_task(self):


        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True



        config = RunConfig.from_file(["/bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/experiment/configs/test/barhaim17_mistral_cs.json"], None)
        train_datasets, test_datasets = collect_datasets(config)
        self.assertEqual(len(test_datasets["fallacy_detection_cmv_adhominem_habernal18"]), 1449)
        self.assertEqual(list(test_datasets.keys()), [
            "counter_argument_generation_cmv_hua18",
            "argument_rating_dagstuhl_15512_effectiveness_wachsmuth17",
            "argument_unit_segmentation_webDiscourse_ajjour17",
            "argument_similarity_ukp_aspect_reimers19",
            "fallacy_detection_cmv_adhominem_habernal18"
        ])


    def test_experiment_setup_corss_task_skill_filtering(self):


        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True



        config = RunConfig.from_file(["/bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/experiment/configs/test/barhaim17_mistral_skill_filtering.json"], None)
        train_datasets, test_datasets = collect_datasets(config)
        self.assertEqual(len(test_datasets["fallacy_detection_cmv_adhominem_habernal18"]), 1449)
        self.assertEqual(list(test_datasets.keys()), [
            "counter_argument_generation_cmv_hua18",
            "argument_rating_dagstuhl_15512_effectiveness_wachsmuth17",
            "argument_unit_segmentation_webDiscourse_ajjour17",
            "argument_similarity_ukp_aspect_reimers19",
            "fallacy_detection_cmv_adhominem_habernal18"
        ])

        self.assertIn("argument_relation_identification_erulemaking_park18",list(train_datasets.keys()))
        self.assertNotIn("aspect_detection_ukp_corpus_schiller21",list(train_datasets.keys()))

    def test_create_in_task_dataset(self):
        config = RunConfig.from_file(["/bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/experiment/configs/hpo/in_task_hpo.json"], None)

        with open(config.experiment_splits_path) as experiment_splits_file:
            experiment_splits = json.load(experiment_splits_file)
            task_path = Path("/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/tasks")
            dataset = create_dataset_in_tasks(task_path, config.model_config.prompt_template,experiment_splits=experiment_splits)

            self.assertEqual(len(list(iter(dataset["test_argument_rating_dagstuhl_15512_effectiveness_wachsmuth17"]))),78)


    def test_create_prompting_dataset(self):
        config = RunConfig.from_file(["/bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/experiment/configs/hpo/in_task_hpo.json"], None)
        task_path = Path("/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/tasks")
        path_dataset = Path("/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/argbench-dataset")
        dataset = create_dataset_prompting(task_path, config.model_config.prompt_template, test_subsample_rate=0.1)


        four_shot_dataset = create_dataset_prompting(task_path, config.model_config.shot_prompt_template, test_subsample_rate=0.1, few_shot_amount=4)
        one_shot_dataset = create_dataset_prompting(task_path, config.model_config.prompt_template, test_subsample_rate=0.1, few_shot_amount=1)
        cot_dataset = create_dataset_prompting(task_path, config.model_config.cot_prompt_template, test_subsample_rate=0.1)
        path = formulate_argbench_dataset_path(ExperimentType.PROMPTING, PromptingTechnique.ZERO_SHOT, sample=True, path_argbench_dataset=path_dataset)
        path_four_shot_dataset = formulate_argbench_dataset_path(ExperimentType.PROMPTING, PromptingTechnique.FOUR_SHOT, sample=True, path_argbench_dataset=path_dataset)
        path_one_shot_dataset = formulate_argbench_dataset_path(ExperimentType.PROMPTING, PromptingTechnique.ONE_SHOT, sample=True, path_argbench_dataset=path_dataset)
        path_cot_dataset = formulate_argbench_dataset_path(ExperimentType.PROMPTING, PromptingTechnique.COT, sample=True, path_argbench_dataset=path_dataset)
        four_shot_dataset.save_to_disk(path_four_shot_dataset)
        one_shot_dataset.save_to_disk(path_one_shot_dataset)

        dataset.save_to_disk(path)
        one_shot_dataset.save_to_disk(path_one_shot_dataset)
        four_shot_dataset.save_to_disk(path_four_shot_dataset)
        cot_dataset.save_to_disk(path_cot_dataset)
        self.assertEqual(len(dataset),57)


    def test_create_argbench_datset(self):
        config = RunConfig.from_file(["/bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/experiment/configs/hpo/in_task_hpo.json"], None)
        path_dataset = Path("/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/argbench-dataset")
        dataset = create_argbench_dataset(ExperimentType.PROMPTING, prompting_technique=PromptingTechnique.COT, sample=True, run_config=config)
        path_cot_dataset = formulate_argbench_dataset_path(ExperimentType.PROMPTING, PromptingTechnique.COT, sample=True,
                                                           path_argbench_dataset=path_dataset)
        self.assertTrue(path_cot_dataset.exists())

    def test_load_argbench_dataset(self):
        config = RunConfig.from_file(["/bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/experiment/configs/hpo/in_task_hpo.json"], None)
        path_dataset = Path("/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/argbench-dataset")
        path_cot_dataset = formulate_argbench_dataset_path(ExperimentType.PROMPTING, PromptingTechnique.COT, sample=False,
                                                           path_argbench_dataset=path_dataset)

        dataset = load_experiment(ExperimentType.PROMPTING, prompting_technique=PromptingTechnique.COT, sample=False, test_task=None, run_config=config)
        self.assertTrue(path_cot_dataset.exists())
        dataset = load_experiment(ExperimentType.PROMPTING, prompting_technique=PromptingTechnique.COT, sample=False, test_task=None, run_config=config)
        split = dataset["test_stance_classification_ibmsc_barhaim17"]
        instance = next(iter(split))
        self.assertIn("think step by ste", instance["input"])
        in_task_dataset = load_experiment(ExperimentType.IN_TASK, prompting_technique=PromptingTechnique.ZERO_SHOT, sample=False, test_task="counter_argument_generation_cmv_hua18", run_config=config)
        self.assertIn("train", in_task_dataset.keys())

    def test_tokenizing_dataset(self):
        config = RunConfig.from_file(["/bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/experiment/configs/hpo/in_task_hpo.json"], None)
        cutoff_len = 200
        path_dataset = Path("/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/argbench-dataset")

        in_task_dataset = load_experiment(ExperimentType.IN_TASK, prompting_technique=PromptingTechnique.ZERO_SHOT, sample=True,
                                          test_task="argument_rating_dagstuhl_15512_effectiveness_wachsmuth17", run_config=config)
        template_formatter = formate_model_template(config.model_config.prompt_template)
        tokenizer = AutoTokenizer.from_pretrained("/bigwork/nhwpajjy/pre-trained-models/DeepSeek-R1-Distill-Qwen-1.5B", padding_side="left",
                                                       truncation=True, max_length = config.cutoff_len,
                                                       trust_remote_code=True
                                                       )
        tokenizer = get_tokenizer(cutoff_len,tokenizer , True)
        in_task_dataset["test"] = in_task_dataset["test"].to_iterable_dataset().map(template_formatter, num_proc=8, load_from_cache_file=True)
        in_task_dataset["test"] = in_task_dataset["test"].to_iterable_dataset().map(tokenizer, num_proc=8, load_from_cache_file=True)