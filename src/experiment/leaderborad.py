import os.path

import pandas as pd
class Leaderboard:

    def __init__(self, output_path):
        self.output_path = output_path


    def read_file(self):
        if os.path.exists(self.output_path):
            self.df_results = pd.read_csv(self.output_path)
        else:
            self.df_results = pd.DataFrame(columns=["model", "training_data", "task", "metric"])

    def add_results(self, results):
        df_record = pd.DataFrame(results)
        self.df_results = pd.concat([self.df_results, df_record])

    def save_file(self,results):
        self.df_results.to_csv(self.output_path, sep="\t")