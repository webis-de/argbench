from unittest import *
from ..leaderborad import *

class testLeaderboard(TestCase):

    def testLeaderBoardAdd(self):
        leaderboard = Leaderboard("/home/yamen/tmp/leaderboard.csv")
        now = datetime.now()
        starting_time = now.strftime("%m-%d-%H:%M:%S")

        metric = {"test_task": "segmentation", "metric":"bleu", "score" : 1/10, "training_data":"MOC", "model": "llama", "start_time":starting_time}
        leaderboard.add_results(metric)
        metric = {"test_task": "segmentation", "metric":"meteor", "score" : 2/10, "training_data":"MOC", "model": "llama", "start_time":starting_time}
        leaderboard.add_results(metric)
        metric = {"test_task": "segmentation", "metric":"rouge", "score" : 3/10, "training_data":"MOC", "model": "llama", "start_time":starting_time}
        leaderboard.add_results(metric)

        leaderboard.save_file()