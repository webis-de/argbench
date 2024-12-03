import pandas as pd
class leaderboard:

    def __init__(self, output_path):
        self.output_path = output_path


    def read_file(self):
        self.df_results = pd.read_csv(self.output_path)

