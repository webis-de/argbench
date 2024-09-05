from common import tasks_path
import json
import pandas as pd


if __name__ == "__main__":
    with open(tasks_path() / "metadata.json", "r") as f:
            metadata = json.load(f)

    data = {
        "Data File": [],
        "Definition Len": [],
        "AVG Instance Len": [],
        "MIN Instance Len": [],
        "MAX Instance Len": [],
        "AVG Positive Len": [],
        "MIN Positive Len": [],
        "MAX Positive Len": [],
        "AVG Negative Len": [],
        "MIN Negative Len": [],
        "MAX Negative Len": []
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
            description_len = len(file_data["Definition"][0])

            if file_data["Instances"]:
                total_instance_len = 0
                max_instance_len = 0
                min_instance_len = 99999999999
                for inst in file_data["Instances"]:
                    instance_len = len(inst["input"])
                    total_instance_len += instance_len
                    if instance_len > max_instance_len:
                        max_instance_len = instance_len
                    if instance_len < min_instance_len:
                        min_instance_len = instance_len
                avg_instance_len = total_instance_len // len(file_data["Instances"])
            else:
                avg_instance_len = 0
                max_instance_len = 0
                min_instance_len = 0

            if file_data["Positive Examples"]:
                total_positive_len = 0
                max_positive_len = 0
                min_positive_len = 99999999999
                for inst in file_data["Positive Examples"]:
                    positive_len = len(inst["input"])
                    total_positive_len += positive_len
                    if positive_len > max_positive_len:
                        max_positive_len = positive_len
                    if positive_len < min_positive_len:
                        min_positive_len = positive_len
                avg_positive_len = total_positive_len // len(file_data["Positive Examples"])
            else:
                avg_positive_len = 0
                max_positive_len = 0
                min_positive_len = 0


            if file_data["Negative Examples"]:
                total_negative_len = 0
                max_negative_len = 0
                min_negative_len = 99999999999
                for inst in file_data["Negative Examples"]:
                    negative_len = len(inst["input"])
                    total_negative_len += negative_len
                    if negative_len > max_negative_len:
                        max_negative_len = negative_len
                    if negative_len < min_negative_len:
                        min_negative_len = negative_len
                avg_negative_len = total_negative_len // len(file_data["Negative Examples"])
            else:
                avg_negative_len = 0
                max_negative_len = 0
                min_negative_len = 0

            data["Data File"].append(file)
            data["AVG Instance Len"].append(avg_instance_len)
            data["MIN Instance Len"].append(min_instance_len)
            data["MAX Instance Len"].append(max_instance_len)
            data["AVG Negative Len"].append(avg_negative_len)
            data["MIN Negative Len"].append(min_negative_len)
            data["MAX Negative Len"].append(max_negative_len)
            data["AVG Positive Len"].append(avg_positive_len)
            data["MIN Positive Len"].append(min_positive_len)
            data["MAX Positive Len"].append(max_positive_len)
            data["Definition Len"].append(description_len)

    df = pd.DataFrame(data)

    print(df.to_markdown())
