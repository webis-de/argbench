from unittest import *
from argbench.experiment.prepare_experiment import *
class ExperimentSetupTest(TestCase):

    def test_dataset_load(self):
        df_test = load_set("stance_classification_ibmsc_barhaim17", "/bigwork/nhwpajjy/computational-argumentation-tasks-instructions", DatasetSplit.TEST)
        df_train = load_set("stance_classification_ibmsc_barhaim17", "/bigwork/nhwpajjy/computational-argumentation-tasks-instructions", DatasetSplit.TRAIN)
        df_val = load_set("stance_classification_ibmsc_barhaim17", "/bigwork/nhwpajjy/computational-argumentation-tasks-instructions", DatasetSplit.VAL)
        df_train_val = load_set("stance_classification_ibmsc_barhaim17", "/bigwork/nhwpajjy/computational-argumentation-tasks-instructions", DatasetSplit.TRAIN_AND_VAL)
        self.assertEqual(len(df_train) + len(df_val), len(df_train_val))


    def test_experiment_setup(self):


        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True



        config = RunConfig.from_file(["/bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/experiment/configs/test/barhaim17_mistral_hpo.json"], None)
        train_datasets, test_datasets = collect_datasets(config)
        self.assertEqual(len(test_datasets["stance_classification_ibmsc_barhaim17"]), 262)
        self.assertEqual(list(test_datasets.keys()), ["stance_classification_ibmsc_barhaim17"])
        self.assertEqual(len(train_datasets["stance_classification_ibmsc_barhaim17"]), 604)