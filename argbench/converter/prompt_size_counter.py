from collections import defaultdict

from common import tasks_path
from statistics import mean
import math
from transformers import LlamaTokenizer, LlamaTokenizerFast, AutoTokenizer
from argparse import ArgumentParser
import json
import pandas as pd


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("/bigwork/nhwpajjy/pre-trained-models/DeepSeek-R1-Distill-Qwen-1.5B", unk_token="<unk>")
        # LlamaTokenizerFast
    with open(tasks_path() / "metadata.json", "r") as f:
            metadata = json.load(f)

    data = {
        "Data File": [],
        "Definition Len": [],
        "AVG Instance Len": [],
        "MIN Instance Len": [],
        "MAX Instance Len": [],
        "AVG Output Len": [],
        "MIN Output Len": [],
        "MAX Output Len": [],
        "MAX Definition + Instance": [],
        "MAX ALL": [],
        "Dataset": [],
        "Count": []
    }
    counter = 0
    for dataset in metadata:
        if "claim" in dataset:
            continue
        for file in metadata[dataset]["file_list"]:
            instances = []
            outputs = []

            file_path = tasks_path() / dataset / file
            try:
                with open(file_path, "r") as f:
                    file_data = json.load(f)
            except FileNotFoundError:
                print(f"File not found, skipping: {file_path}")
                continue

            print("------------------------------")
            print(file)

            if not file_data["Instances"]:
                print(f"File has no instances: {file_path}")
                continue
            for inst in file_data["Instances"]:
                if not inst["input"] or not inst["output"] or isinstance(inst["input"], float) or isinstance(inst["output"], float):
                    continue
                instances.append(inst["input"])

                outputs.append(inst["output"][0])


            definition = file_data["Definition"][0]

            del file_data

            if tokenizer:
                instances = [tokenizer(text=inst, return_tensors="pt")["input_ids"][0] for inst in instances]
                outputs = [tokenizer(text=out, return_tensors="pt")["input_ids"][0] for out in outputs]
                definition = tokenizer(text=definition, return_tensors="pt")["input_ids"][0]

            description_len = len(definition)

            max_instance_len = max((len(inst) for inst in instances))
            min_instance_len = min((len(inst) for inst in instances))
            avg_instance_len = mean((len(inst) for inst in instances))

            max_outputs_len = max((len(out) for out in outputs))
            min_outputs_len = min((len(out) for out in outputs))
            avg_outputs_len = mean((len(out) for out in outputs))

            max_def_inst = max_instance_len + description_len
            max_all = max_def_inst + max_outputs_len

            print(description_len)
            print(avg_instance_len)
            print(max_instance_len)
            print(min_instance_len)
            data["Dataset"].append(dataset)
            data["Data File"].append(file)
            data["Count"].append(len(instances))
            data["AVG Instance Len"].append(avg_instance_len)
            data["MIN Instance Len"].append(min_instance_len)
            data["MAX Instance Len"].append(max_instance_len)
            data["AVG Output Len"].append(avg_outputs_len)
            data["MIN Output Len"].append(min_outputs_len)
            data["MAX Output Len"].append(max_outputs_len)
            data["Definition Len"].append(description_len)
            data["MAX Definition + Instance"].append(max_def_inst)
            data["MAX ALL"].append(max_all)
        df = pd.DataFrame(data)
        dataset_records = {}
        counter += 1
        if counter == 2:
            break

    for column in df.columns:
        if "MIN" in column:
            dataset_records[column]= df.groupby("Dataset").agg({column: "min"})[column].values.tolist()
        elif "MAX" in column:
            dataset_records[column] = df.groupby("Dataset").agg({column: "max"})[column].values.tolist()
        elif column == "Data File":
            dataset_records[column]= df.groupby("Dataset").agg({"Dataset": lambda x: list(set(x))})["Dataset"].apply(lambda x: x[0]).values.tolist()
        elif column == "Dataset":
            dataset_records[column]= df.groupby("Dataset").agg({column: pd.DataFrame.sample})[column].values.tolist()
        elif column == "Count":
            dataset_records[column]= df.groupby("Dataset").agg({column: "sum"})[column].values.tolist()
        else:
            dataset_records[column]= df.groupby("Dataset").agg({column: "mean"})[column].values.tolist()
    df = pd.concat([df, pd.DataFrame(dataset_records)])
    df.to_csv("/bigwork/nhwpajjy/benchmark-count.csv", index=False, float_format ="%.2f")
