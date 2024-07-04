#!/usr/bin/env python3
from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq, GenerationConfig
from peft import PeftModel, prepare_model_for_kbit_training
from preprocess import collect_datasets, tasks_path, get_metadata
from tqdm import tqdm
from sklearn.metrics import f1_score
import torch

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

tokenizer = LlamaTokenizer.from_pretrained(base_model)
tokenizer.pad_token_id = (0)

print("tokenizer loaded")
model = LlamaForCausalLM.from_pretrained(
    base_model,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map=device_map
)
model = prepare_model_for_kbit_training(model)

print("model loaded")
model = PeftModel.from_pretrained(model, lora_model)
print("LoRa loaded")

datasets_path = tasks_path()
metadata = get_metadata()
_, test_instances = collect_datasets(config, metadata, datasets_path)

generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
)

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

predictions = []
for i, data in tqdm(enumerate(test_instances)):
    prompt = tokenizer(data["input"], return_tensors="pt")
    inputs = prompt["input_ids"].cuda()
    generated = model.generate(input_ids=inputs, generation_config=generation_config, return_dict_in_generate=True, output_scores=True, max_new_tokens=3)
    output = tokenizer.decode(generated.sequences[0])
    predictions.append(output)

print(compute_f1_score(predictions, [d["output"] for d in test_instances]))
