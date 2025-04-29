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

    def test_experiment_setup_corss_task(self):


        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True



        config = RunConfig.from_file(["/bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/experiment/configs/test/barhaim17_mistral_hpo_cs.json"], None)
        train_datasets, test_datasets = collect_datasets(config)
        self.assertEqual(len(test_datasets["stance_classification_ibmsc_barhaim17"]), 1355)
        self.assertEqual(list(test_datasets.keys()), ["stance_classification_ibmsc_barhaim17"])
