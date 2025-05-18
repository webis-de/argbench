import pandas as pd
from argparse import *
import json
parser = ArgumentParser()
parser.add_argument("--file", type=str)
parser.add_argument("--metadata", type=str)
parser.add_argument("--k", type=int)
parser.add_argument("--cot", action='store_true')

args= parser.parse_args()
df = pd.read_csv(args.file, sep="\t")
df.drop_duplicates(["test_task","k","model","score"],inplace=True)
with open(args.metadata) as file:
    metadata = json.load(file)
    df = df[df["test_task"]!="generation"]
    df = df[df["test_task"]!="quality-assessment"]
    df = df[df["test_task"]!="reasoning"]
    df = df[df["test_task"]!="reasonableness_scoring_cmv_habernal18"]
    df = df[df["test_task"]!="perspective-assessment"]
    df = df[df["test_task"]!="mining"]
    if args.k:
        df = df[df["k"] == args.k]
    elif args.cot:
        df = df[df["model"].str.contains("cot")]
    else:
        df = df[~df["model"].str.contains("cot")]
        df = df[df["k"] != 1]
        df = df[df["k"] != 4]
    print(f"found tasks {len(df)}")
    num_tasks = df["test_task"].nunique()
    print(f"found {num_tasks} tasks")
    df["skill"] = df["test_task"].apply(lambda x :metadata[x]["skill"])
    df.sort_values(by="skill", inplace = True)
    for skill, df_skill in df.groupby("skill"):
        print(skill)
        if skill == "generation":
            df_blue_records = df_skill[df_skill["metric"]=="bleu"]
            blue_score_agg =df_blue_records["score"].mean()
            print(f"bleu {blue_score_agg:.2f}")
            df_bertscore_records = df_skill[df_skill["metric"]=="bertscore"]
            blue_score_agg =df_bertscore_records["score"].mean()
            bertscore = (df_bertscore_records["score"].mean())
            print(f"bertscore {bertscore:.2f}")

        else:
            df_fscore_records = df_skill[df_skill["metric"]=="fscore"]
            fscore_agg =df_fscore_records["score"].mean()
            print(f"fscore {fscore_agg:.2f}")
