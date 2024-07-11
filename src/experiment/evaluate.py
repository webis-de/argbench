#!/usr/bin/env python3
from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq, GenerationConfig
from peft import PeftModel, prepare_model_for_kbit_training
from preprocess import collect_datasets, tasks_path, get_metadata
from tqdm import tqdm
from sklearn.metrics import f1_score
from argparse import ArgumentParser
import torch

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

config = {
    "seed": 42,
    "train_datasets": {},
    "test_datasets": {
        "barhaim21_key_point": {
            "match": "barhaim21_key_point",
            "prompt_template": "{definition}\nPositive Example:\nInput: {positive_example_input_0}\nOutput: {positive_example_output_0}\nNegative Example:\nInput: {negative_example_input_0}\nOutput: {negative_example_output_0}\nInput: {instance_input}\n Output:",
            "subsample_rate": 0.1
        }
    }
}

base_model = "/bigwork/nhwpbozd/decapoda-research-llama-7B-hf/"
lora_models = {
    "adapter_1": "/bigwork/nhwpbozd/ibmsc/checkpoint-200/",
    "adapter_2":"/bigwork/nhwpbozd/ajjour-19-webis-argument-framing/checkpoint-200/"
}
adapter_weights = [
    0.5,
    0.5
]
combination_type = "linear"
device_map = "auto"

datasets_path = tasks_path()
metadata = get_metadata()
_, test_instances = collect_datasets(config, metadata, datasets_path)

print(len(test_instances))

generation_config = GenerationConfig(
    # temperature=0.1,
    top_p=0.75,
    length_penalty=2.0,
    # top_k=40,
    num_beams=1,
)

if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate PEFT adapters")
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
    lora_mapping = list(lora_models.items())
    first_model = lora_mapping[0]
    model = PeftModel.from_pretrained(model, first_model[1], first_model[0])

    if len(lora_models) > 1:
        for adapter in lora_mapping[1:]:
            model.load_adapter(adapter[1], adapter[0])
    print("LoRa loaded")

    if len(lora_models) > 1:
        model.add_weighted_adapter(
            [a_n for a_n in lora_models],
            adapter_weights,
            "full_adapter",
            combination_type
        )

    predictions = []
    for i, data in tqdm(enumerate(test_instances)):
        prompt = tokenizer(data["input"], return_tensors="pt")
        inputs = prompt["input_ids"].cuda()
        generated = model.generate(input_ids=inputs, generation_config=generation_config, return_dict_in_generate=True, output_scores=True, max_new_tokens=3)
        output = tokenizer.decode(generated.sequences[0])
        print(output)
        predictions.append(output)

    print(compute_f1_score(predictions, [d["output"] for d in test_instances]))
