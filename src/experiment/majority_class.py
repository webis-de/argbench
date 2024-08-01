from preprocess import collect_datasets, tasks_path, get_metadata
from sklearn.metrics import f1_score
from collections import Counter

base_model = "/bigwork/nhwpbozd/decapoda-research-llama-7B-hf/"
lora_model = "/bigwork/nhwpbozd/train-checkpoint/checkpoint-200/"
device_map = "auto"

config = {
    "seed": 42,
    "train_datasets": {
        "stab18_stance_classification": {
            "match": "stab18_stance_classification",
            "prompt_template": "{definition}\nPositive Example:\nInput: {positive_example_input_0}\nOutput: {positive_example_output_0}\nNegative Example:\nInput: {negative_example_input_0}\nOutput: {negative_example_output_0}\nInput: {instance_input}\n Output:"
        }
    },
    "test_datasets": {
        "barhaim21_key_point": {
            "match": "barhaim21_key_point",
            "prompt_template": "{definition}\nPositive Example:\nInput: {positive_example_input_0}\nOutput: {positive_example_output_0}\nNegative Example:\nInput: {negative_example_input_0}\nOutput: {negative_example_output_0}\nInput: {instance_input}\n Output:"
        }
    }
}

def compute_f1_score(predictions, references, f1_average="macro"):
    label_mapping = {}
    label_targets = []
    label_predictions = []
    for i in range(len(predictions)):
        ground_truth = references[i]
        predicted = predictions[i]
        if ground_truth not in label_mapping:
            label_mapping[ground_truth] = len(label_mapping) + 1
        if predicted not in label_mapping:
            label_mapping[predicted] = len(label_mapping) + 1
        label_targets.append(label_mapping[ground_truth])
        label_predictions.append(label_mapping[predicted])


    return f1_score(label_targets, label_predictions, average=f1_average)

datasets_path = tasks_path()
metadata = get_metadata()
_, test_instances = collect_datasets(config, metadata, datasets_path)

labels = Counter()

for data in test_instances:
    labels[data["output"]] += 1

most_common_label = labels.most_common(1)[0][0]
predictions = [most_common_label] * len(test_instances)

print(compute_f1_score(predictions, [d["output"] for d in test_instances]))
