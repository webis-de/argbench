from unittest import *
from argbench.experiment.leaderborad import *

class testLeaderboard(TestCase):

    def testLeaderBoardAdd(self):
        leaderboard = Leaderboard("/home/yamen/tmp/leaderboard.csv")
        now = datetime.now()
        starting_time = now.strftime("%m-%d-%H:%M:%S")

        metric = {"test_task": "argument_relation_identification_erulemaking_park18", "metric":"bleuscore", "score" : 1/10, "training_data":"MOC", "model": "llama", "start_time":starting_time, "k":10}
        leaderboard.add_results(metric)
        metric = {"test_task": "argument_relation_identification_erulemaking_park18", "metric":"fscore", "score" : 2/10, "training_data":"MOC", "model": "llama", "start_time":starting_time, "k":10}
        leaderboard.add_results(metric)
        metric = {"test_task": "argument_rating_dagstuhl_15512_overall_quality_wachsmuth17", "metric":"fscore", "score" : 3/10, "training_data":"MOC", "model": "llama", "start_time":starting_time, "k":10}
        leaderboard.add_results(metric)

        leaderboard.save_file()