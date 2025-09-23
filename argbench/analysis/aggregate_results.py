import pandas as pd
from argparse import *
import json
parser = ArgumentParser()
parser.add_argument("--file", type=str)
parser.add_argument("--metadata", type=str)
parser.add_argument("--k", type=int)
parser.add_argument("--cot", action='store_true')
parser.add_argument("--missing-tasks", action='store_true')
parser.add_argument("--skill-output-path", type=str)
parser.add_argument("--seed", type=int)
parser.add_argument("--last-run-per-task", action="store_true")
args= parser.parse_args()
df = pd.read_csv(args.file, sep="\t")
#df.drop_duplicates(["test_task","k","model","score"],inplace=True)
with open(args.metadata) as file:

    metadata = json.load(file)
    df = df[df["seed"]==args.seed]
    df = df[df["test_task"]!="generation"]
    df = df[df["test_task"]!="quality-assessment"]
    df = df[df["test_task"]!="reasoning"]
    df = df[df["test_task"]!="reasonableness_scoring_cmv_habernal18"]
    df = df[df["test_task"]!="perspective-assessment"]
    df = df[df["test_task"]!="mining"]
    processed_tasks = set(df["test_task"].tolist())
    expected_tasks = metadata.keys()
    missing_tasks = expected_tasks - processed_tasks
    print("** Missing Tasks **\n")
    print(list(missing_tasks).sort())
    print("\n**               **\n")
    path_output = args.skill_output_path
    prompting_technique=""
    if args.k:
        df = df[df["k"] == args.k]
        prompting_technique=f"few-shot-{args.k}"
    elif args.cot:
        df = df[df["model"].str.contains("cot")]
        prompting_technique=f"cot"
    else:
        df = df[~df["model"].str.contains("cot")]
        df = df[df["k"] != 1]
        df = df[df["k"] != 4]
        prompting_technique=f"few-shot-{0}"
    if not args.last_run_per_task:
        times = sorted(df["start_time"].values)
        last_time = times[-1]

        print(f"\n** aggregating run on {last_time}**\n")
        df = df[df["start_time"]==last_time]
        last_run_per_task = False
    else:
        last_run_per_task = True
    experiment = df["model"].iloc[0]
    print(f"found tasks {len(df)}")
    num_tasks = df["test_task"].nunique()
    print(f"found {num_tasks} tasks")
    print("\n-------------\n")
    df["skill"] = df["test_task"].apply(lambda x :metadata[x]["skill"])
    skills = ["mining", "perspective-assessment", "quality-assessment", "reasoning", "generation"]
    scores = 0
    df["skill-index"] = df["skill"].apply(lambda x: skills.index(x) )
    df.sort_values(by="skill-index", inplace=True)
    for skill, df_skill in df.groupby("skill", sort=False):
        task_count = df_skill['test_task'].nunique()
        print(f"\n-----overview ({skill})----(count {task_count})---\n")
        if not last_run_per_task:
            df_skill_overview = df_skill[df_skill["metric"].isin(["generation-score", "fscore"])]
            df_skill_overview = df_skill_overview[["test_task", "score"]]
            if not last_run_per_task:
                df_skill_overview.to_csv(f"{path_output}/{skill}-{experiment}-{prompting_technique}-{last_time}-{args.seed}.csv", index=False)
            else:
                df_skill_overview.to_csv(f"{path_output}/{skill}-{experiment}-{prompting_technique}-{args.seed}.csv", index=False)
            print(df_skill_overview)
            print(f"--------------------\n")
        if skill == "generation":
            df_blue_records = df_skill[df_skill["metric"]=="bleu"]
            if last_run_per_task:
                df_blue_records.sort_values(["test_task", "start_time"], ascending=False, inplace=True)
                df_blue_records = df_blue_records.groupby("test_task").first()
            blue_score_agg =df_blue_records["score"].mean()

            print(f"{skill:<30} bleu {blue_score_agg:>14.2f}")
            df_bertscore_records = df_skill[df_skill["metric"]=="bertscore"]
            if last_run_per_task:
                df_bertscore_records.sort_values(["test_task", "start_time"], ascending=False, inplace=True)
                df_bertscore_records = df_bertscore_records.groupby("test_task").first()

            bertscore = (df_bertscore_records["score"].mean())
            generation_score = (bertscore + blue_score_agg) /2
            print(f"{skill:<30} generationscore {generation_score:>9.2f}")
            print(f"{skill:<30} bertscore {bertscore:>9.2f}\n----------\n")



            scores += generation_score
        else:
            df_fscore_records = df_skill[df_skill["metric"]=="fscore"]
            if last_run_per_task:
                df_fscore_records.sort_values(["test_task", "start_time"], ascending=False, inplace=True)
                df_fscore_records = df_fscore_records.groupby("test_task").first()

            fscore_agg =df_fscore_records["score"].mean()
            print(f"{skill:<30} fscore {fscore_agg:>12.2f}\n-------------\n")
            scores += fscore_agg
    all = scores / 5
    all_skill = "all"
    print(f"{all_skill:<30} macro {all:>13.2f}\n-------------\n")
