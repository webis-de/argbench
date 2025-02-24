from common import tasks_path
from statistics import mean
from transformers import LlamaTokenizer, LlamaTokenizerFast
from argparse import ArgumentParser
import json
import pandas as pd


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Count length of each dataset")

    arg_parser.add_argument("-tn", "--tokenizer_name", type=str, help="Tokenizer name to use")
    arg_parser.add_argument("-tt", "--tokenizer_type", choices=["llama_tokenizer"], help="Type of tokenizer to use")

    args = arg_parser.parse_args()

    if args.tokenizer_type == "llama_tokenizer":
        tokenizer = LlamaTokenizerFast.from_pretrained(args.tokenizer_name, unk_token="<unk>")
        # LlamaTokenizerFast
    else:
        tokenizer = None

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
        "MAX ALL": []
    }

    for dataset in metadata:
        for file in metadata[dataset]["file_list"]:
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

            instances = [inst["input"] for inst in file_data["Instances"]]
            outputs = [inst["output"][0] for inst in file_data["Instances"]]

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

            data["Data File"].append(file)
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

    print(df.to_markdown())
