from unittest import *
from ..leaderborad import *

class testLeaderboard(TestCase):

    def testLeaderBoardAdd(self):
        leaderboard = Leaderboard("/home/yamen/tmp/leaderboard.csv")

        for i in range(10):
            metric = {"test_task": "segmentation", "metric":"bleu", "score" : i/10, "training_data":"MOC", "model": "llama"}
            leaderboard.add_results(metric)
            metric = {"test_task": "segmentation", "metric":"meteor", "score" : i*2/10, "training_data":"MOC", "model": "llama"}
            leaderboard.add_results(metric)
            metric = {"test_task": "segmentation", "metric":"rouge", "score" : i*3/10, "training_data":"MOC", "model": "llama"}
            leaderboard.add_results(metric)

        leaderboard.save_file()