import os.path
from unittest import *
from ..leaderborad import *
from ..hpo_output import *
class testLeaderboard(TestCase):

    def testHPOBoardAdd(self):

        hpo_output = HPOOutput("/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/hpo-results.csv")
        now = datetime.now()
        starting_time = now.strftime("%m-%d-%H:%M:%S")

        metric = {"test_task": "segmentation", "metric":"bleu", "score" : 1/10, "experiment":"MOC", "model": "llama", "start_time":starting_time}
        hpo_output.add_results(metric)
        metric = {"test_task": "classification", "metric":"meteor", "score" : 2/10, "experiment":"MOC", "model": "llama", "start_time":starting_time}
        hpo_output.add_results(metric)
        metric = {"test_task": "regression", "metric":"rouge", "score" : 3/10, "experiment":"MOC", "model": "llama", "start_time":starting_time}
        hpo_output.add_results(metric)

        hpo_output.save_file()
        self.assertTrue(os.path.exists("/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/hpo-results.csv"))