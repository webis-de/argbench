#!/usr/bin/env python3
from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import PeftModel, prepare_model_for_kbit_training
from preprocess import collect_datasets, tasks_path, get_metadata
import torch

base_model = "/bigwork/nhwpajjy/pre-trained-models/llama-7b-hf"
lora_model = "/bigwork/nhwpajjy/pre-trained-models/alpaca-lora-7b/"
device_map = "auto"

config = {
    "seed": 42,
    "train_datasets": {
        "stab18_stance_classification": {
            "match": "stab18_stance_classification",
            "prompt_template": "### Instruction:\n{definition}\n### Input:\nPositive Example:\nInput: {positive_example_input_0}\nOutput: {positive_example_output_0}\nNegative Example:\nInput: {negative_example_input_0}\nOutput: {negative_example_output_0}\nInput: {instance_input}\n### Response:"
        }
    },
    "test_datasets": {
        "barhaim21_key_point": {
            "match": "barhaim21_key_point",
            "prompt_template": "### Instruction:\n{definition}\n### Input:\nPositive Example:\nInput: {positive_example_input_0}\nOutput: {positive_example_output_0}\nNegative Example:\nInput: {negative_example_input_0}\nOutput: {negative_example_output_0}\nInput: {instance_input}\n### Response:"
        }
    }
}

cutoff_len=1500
batch_size = 32
warmup_steps = 0
num_epochs = 2
learning_rate = 3e-4
optim = "adamw_torch"
output_dir = "/bigwork/nhwpbozd/train-checkpoint"

tokenizer = LlamaTokenizer.from_pretrained(base_model)
tokenizer.pad_token_id = (0)

print("tokenizer loaded")
model = LlamaForCausalLM.from_pretrained(
    base_model,
    load_in_4bit=True,
    # load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map=device_map
)
model = prepare_model_for_kbit_training(model)
model.enable_input_require_grads()

print("model loaded")
model = PeftModel.from_pretrained(model, lora_model, is_trainable=True)
print("LoRa loaded")

datasets_path = tasks_path()
metadata = get_metadata()
train_instances, test_instances = collect_datasets(config, metadata, datasets_path)

model.print_trainable_parameters()
print("data loaded")


def tokenize(prompt, add_eos_token=True):
    result = tokenizer(prompt, truncation=True, max_length=cutoff_len, padding=False, return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


def generate_and_tokenize_prompt(data_point, train=True):
    input_prompt = tokenize(data_point['input'])
    if train:
        full_prompt = tokenize(f"{data_point['input']}{data_point['output']}")
        instruction_len = len(input_prompt) - 1
        full_prompt["labels"] = [
            -100
        ] * instruction_len + full_prompt["labels"][
            instruction_len:
        ]
        return full_prompt
    return input_prompt

train_data = [generate_and_tokenize_prompt(d) for d in train_instances]
test_data = [generate_and_tokenize_prompt(d) for d in test_instances]

print("data processed")

train_args = TrainingArguments(
    per_device_train_batch_size=batch_size,
    warmup_steps=warmup_steps,
    num_train_epochs=num_epochs,
    learning_rate=learning_rate,
    fp16=True,
    logging_steps=10,
    optim=optim,
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=200,
    save_steps=200,
    output_dir=output_dir,
    save_total_limit=3
)

trainer = Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=test_data,
    args=train_args,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)

trainer.train()

model.save_pretrained(output_dir)
