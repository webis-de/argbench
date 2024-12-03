import os.path

import pandas as pd
from datetime import datetime

class Leaderboard:

    def __init__(self, output_path):
        self.output_path = output_path


    def read_file(self):
        if os.path.exists(self.output_path):
            self.df_results = pd.read_csv(self.output_path)
        else:
            self.df_results = pd.DataFrame(columns=["model", "training_data", "test-task", "metric", "time"])

    def add_results(self, results):
        df_record = pd.DataFrame(results)
        now = datetime.now()
        time_now = now.strftime("%m-%d-%H:%M:%S")
        results["time"] = time_now

        self.df_results = pd.concat([self.df_results, df_record])

    def save_file(self):
        self.df_results.to_csv(self.output_path, sep="\t")