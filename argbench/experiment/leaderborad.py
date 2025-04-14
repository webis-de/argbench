import os.path

import pandas as pd
from datetime import datetime



from argbench.experiment.utils import get_metadata


class Leaderboard:
    """
    This class keeps track of LLM models' score for one experiment for example leave five tasks evaluation
    """
    def __init__(self, output_path):
        self.output_path = output_path
        self.read_file()
        self.added_results = []
    def read_file(self):
        if os.path.exists(self.output_path):
            self.df_results = pd.read_csv(self.output_path, sep="\t")
        else:
            self.df_results = pd.DataFrame(columns=["model",  "test_task",  "metric", "score", "start_time", "k", "filter"])


    def add_aggregated_results(self):
        metadata = get_metadata()
        task_skill_mapping = {task: metadata[task]["skill"] for task in metadata}

        skill_results = {skill: 0 for skill in task_skill_mapping.values()}
        skill_counts = {skill: 0 for skill in task_skill_mapping.values()}
        skill_records = {}


        for task_result in self.added_results:
            test_task = task_result["test_task"]
            skill = task_skill_mapping[test_task]
            if task_result["metric"] == "fscore" or task_result["metric"]== "bertscore":
                skill_results[skill] += task_result["score"]
                skill_counts[skill] += 1
                skill_records[skill]={}

        if self.added_results:
            model = self.added_results[0]["model"]
            start_time = self.added_results[0]["start_time"]
            k = self.added_results[0]["k"]

        for skill in skill_records:
            score = skill_results[skill] / skill_counts[skill]
            if skill == "generation":
                metric = "bertscore"
            else:
                metric =  "fscore"
            skill_records[skill] = {"model": model, "start_time": start_time, "k": k, "score": score, "metric": metric, "test_task": skill}
            df_record = pd.DataFrame([skill_records[skill]])
            self.df_results = pd.concat([self.df_results, df_record])

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
        self.added_results.append(results)
        self.df_results = pd.concat([self.df_results, df_record])

    def pivot(self):

        self.df_results["metric"] = self.df_results.apply(lambda record: record["test_task"]+"_"+record["metric"],axis=1)
        pivoted_df = self.df_results.pivot(index=["model",  "start_time"],values="score",columns="metric").reset_index()


        return pivoted_df

    def save_file(self):
        self.add_aggregated_results()
        self.df_results.to_csv(self.output_path, sep="\t", index=False)
        pivated_results = self.pivot()
        pivated_results.to_csv(self.output_path.replace(".csv", "-pivoted.csv"), sep="\t", index=False)
