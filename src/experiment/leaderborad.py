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
            self.df_results = pd.read_csv(self.output_path, sep="\t")
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

        now = datetime.now()
        time_now = now.strftime("%m-%d-%H:%M:%S")
        results["time"] = time_now
        df_record = pd.DataFrame([results])
        self.df_results = pd.concat([self.df_results, df_record])

    def pivot(self):
        all_data_frames =  []
        for _, df_results_task in self.df_results.groupby("test_task"):
            pivoted_df = df_results_task.pivot(index=["model","time","training_data"],values=["score"],columns="metric")
            all_data_frames.append(pivoted_df)
        return pd.concat(all_data_frames)

    def save_file(self):
        pivated_results = self.pivot()
        pivated_results.to_csv(self.output_path, sep="\t", index=False)