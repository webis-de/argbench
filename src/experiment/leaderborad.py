import os.path

import pandas as pd
from datetime import datetime

class Leaderboard:
    """
    This class keeps track of LLM models' score for one experiment for example leave five tasks evaluation
    """
    def __init__(self, output_path):
        self.output_path = output_path
        self.read_file()

    def read_file(self):
        if os.path.exists(self.output_path):
            self.df_results = pd.read_csv(self.output_path)
        else:
            self.df_results = pd.DataFrame(columns=["model", "training_data", "test_task", "metric", "score", "time"])

    def add_results(self, results):
        """
        added
        :param results: a dictionary that contains
        model name which is a string
        training_data: on which data is the model trained
        test_task: the test task to evaluate the model on
        metric: is a float number the model scored on the test task
        time: the time when the experiment finished
        :return:
        """
        df_record = pd.DataFrame([results])
        now = datetime.now()
        time_now = now.strftime("%m-%d-%H:%M:%S")
        results["time"] = time_now

        self.df_results = pd.concat([self.df_results, df_record])

    def save_file(self):
        self.df_results.to_csv(self.output_path, sep="\t", index=False)