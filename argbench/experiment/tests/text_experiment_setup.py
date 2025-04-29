from unittest import *
from argbench.experiment.prepare_experiment import *
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
        self.assertIs([
            "warrant_generation_art_chakarbarty21",
            "argument_rating_dagstuhl_15512_overall_quality_wachsmuth17",
            "argument_relation_identification_microtexts_2_skeppstedt18",
            "stance_classification_ukp_sentential_stab18",
            "fallacy_detection_logic_jin22"
        ],train_datasets.keys())


    def test_experiment_setup_corss_task_skill_filtering(self):


        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True



        config = RunConfig.from_file(["/bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/experiment/configs/test/barhaim17_mistral_cs_skill_filter.json"], None)
        train_datasets, test_datasets = collect_datasets(config)
        self.assertEqual(len(test_datasets["fallacy_detection_cmv_adhominem_habernal18"]), 1449)
        self.assertEqual(list(test_datasets.keys()), [
            "counter_argument_generation_cmv_hua18",
            "argument_rating_dagstuhl_15512_effectiveness_wachsmuth17",
            "argument_unit_segmentation_webDiscourse_ajjour17",
            "argument_similarity_ukp_aspect_reimers19",
            "fallacy_detection_cmv_adhominem_habernal18"
        ])

        self.assertIn("argument_relation_identification_erulemaking_park18",list(train_datasets.keys())
        )
