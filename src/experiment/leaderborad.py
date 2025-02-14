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
            self.df_results = pd.DataFrame(columns=["model",  "test_task",  "metric", "score", "start_time", "k"])

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
        df_record = pd.DataFrame([results])

        self.df_results = pd.concat([self.df_results, df_record])

    def pivot(self):

        self.df_results["metric"] = self.df_results.apply(lambda record: record["test_task"]+"_"+record["metric"],axis=1)
        pivoted_df = self.df_results.pivot(index=["model",  "start_time"],values="score",columns="metric").reset_index()


        return pivoted_df

    def save_file(self):
        self.df_results.to_csv(self.output_path, sep="\t", index=False)
        pivated_results = self.pivot()
        pivated_results.to_csv(self.output_path.replace(".csv", "-pivoted.csv"), sep="\t", index=False)
