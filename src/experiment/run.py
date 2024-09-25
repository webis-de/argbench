#!/usr/bin/env python3
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from pathlib import Path
from dataclasses import asdict
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaTokenizerFast,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    EarlyStoppingCallback,
    BitsAndBytesConfig
)
from peft import (
    PeftModel,
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)
from preprocess import collect_datasets, tasks_path, get_metadata, PandasDataset
from tqdm import tqdm
from config import RunConfig
from evaluate import compute_precision_recall_fscore_support, compute_rouge_score, compute_bleu_score
import json
import torch

def eval_collate(batch):
    out_batch = {k: [] for k in batch[0]}

    for b in batch:
        for k in b:
            out_batch[k].append(b[k])

    return out_batch

class Runner:
    """Model runner class"""

    def __init__(self, config: RunConfig):
        self.config = config
        self.tokenizer = LlamaTokenizerFast.from_pretrained(config.base_model, padding_side="left", unk_token="<unk>")
        self.tokenizer.pad_token_id = config.pad_token_id
        self.model = self.prepare_llama_for_causal_llm(
            config.base_model
        )
        if config.is_eval:
            self.model = prepare_model_for_kbit_training(self.model)
            self.model.enable_input_require_grads()
        print("Model loaded")
        if config.peft_configs and config.peft_fresh_config:
            raise RuntimeError("Cannot instantiate both fresh and trained models")
        if config.peft_configs:
            self.model = self.prepare_peft_model(self.model)
        if config.peft_fresh_config:
            self.model = self.prepare_new_peft_model(self.model)
        print("LoRa loaded")

        self.train_instances, self.test_instances = self.prepare_data()

        if (not self.config.is_eval) and self.config.training_args_config:
            self.trainer = self.prepare_trainer()
        else:
            self.trainer = None

        self.generation_config = GenerationConfig(**config.generation_config.to_conf())


    def prepare_trainer(self):
        train_args = TrainingArguments(**self.config.training_args_config.to_conf())

        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            **self.config.data_collator_config.to_conf()
        )

        callbacks = []

        if config.early_stopping_config:
            callbacks = [
                EarlyStoppingCallback(**self.config.early_stopping_config.to_conf())
            ]

        train_data = []
        for row in self.train_instances.iterrows():
            row = row[1]
            processed = self.generate_and_tokenize_prompt(row, config.cutoff_len)
            train_data.append(processed)

        test_data = []
        for row in self.test_instances.iterrows():
            row = row[1]
            processed = self.generate_and_tokenize_prompt(row, config.cutoff_len, False)
            test_data.append(processed)

        return Trainer(
            model=self.model,
            callbacks=callbacks,
            train_dataset=train_data,
            eval_dataset=test_data,
            args=train_args,
            data_collator=data_collator,
        )

    def prepare_data(self):
        return collect_datasets(
            self.config
        )


    def prepare_llama_for_causal_llm(self, base_model):
        quant_conf = BitsAndBytesConfig(**self.config.quant_config.to_conf())
        return LlamaForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            quantization_config=quant_conf,
            **self.config.llama_causal_config.to_conf()
        )


    def prepare_peft_model(self, model):
        """
        Prepare one or many peft models from pretrained weights
        """
        if len(self.config.peft_configs) > 1:
            first_config = self.config.peft_configs[0]
            # model.load_adapter(first_config.model_id, first_config.adapter_name)
            # model = PeftMixedModel.from_pretrained(model, **first_config.to_conf())
            model = PeftModel.from_pretrained(model, **first_config.to_conf())
            for adapter in self.config.peft_configs[1:]:
                model.load_adapter(adapter.model_id, adapter.adapter_name)

            model.add_weighted_adapter(
                [c.adapter_name for c in self.config.peft_configs],
                [c.adapter_weight for c in self.config.peft_configs],
                "full_adapter",
                self.config.combination_type
            )
            return model

        model = PeftModel.from_pretrained(model, **self.config.peft_configs[0].to_conf())
        return model


    def prepare_new_peft_model(self, model):
        """
        Prepares new peft model
        """
        if adapter_config.adapter_type == "lora":
            config = LoraConfig(**self.config.peft_fresh_config.config_args)
        else:
            raise RuntimeError(f"No such adapter type: {self.config.peft_fresh_config.adapter_type}")

        return get_peft_model(model, config, self.config.peft_fresh_config.adapter_name)



    def tokenize(self, prompt, cutoff_len, add_eos_token=True):
        result = self.tokenizer(prompt, truncation=True, max_length=cutoff_len, padding=False, return_tensors=None,
        )
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        print(result["input_ids"][0].shape)

        return result


    def generate_and_tokenize_prompt(self, data_point, cutoff_len, train=True):
        input_prompt = self.tokenize(data_point['input'], cutoff_len)
        if train:
            full_prompt = self.tokenize(f"{data_point['input']}{data_point['output']}", cutoff_len)
            instruction_len = len(input_prompt) - 1
            full_prompt["labels"] = [
                -100
            ] * instruction_len + full_prompt["labels"][
                instruction_len:
            ]
            return full_prompt
        return input_prompt


    def train(self):
        if self.trainer:
            print("Start training")
            self.trainer.train()


    def write_run(self, run_results):
        with open(self.config.run_config_path, "w") as f:
            run_data = {
                "config": asdict(self.config),
                "run_sesults": run_results
            }
            json.dump(run_data)


    def evaluate(self):
        labels = self.test_instances["output"]
        dataset = PandasDataset(self.test_instances)
        predictions = []

        loader = DataLoader(
            dataset,
            batch_size=self.config.validation_config.batch_size,
            shuffle=True,
            collate_fn=eval_collate,
            pin_memory=True,
            num_workers=6
        )

        for data in tqdm(loader):
            text = data["input"]
            prompt = self.tokenizer(text=text, return_tensors="pt", padding=True)
            inputs = prompt["input_ids"].cuda()
            generated = self.model.generate(
                input_ids=inputs,
                generation_config=self.generation_config,
                return_dict_in_generate=True
            )
            output = self.tokenizer.batch_decode(generated.sequences, skip_special_tokens=True)
            # output = self.tokenizer.decode(gen_diff[0])
            output = [o[len(text[i]):] for i, o in enumerate(output)]
            predictions += output

        if self.config.validation_config.eval_metric == "fscore":
            return compute_precision_recall_fscore_support(
                predictions,
                labels,
                f1_average=self.config.validation_config.fscore_average,
                beta=self.config.validation_config.fscore_beta
            )
        elif self.config.validation_config.eval_metric == "rouge":
            return compute_rouge_score(predictions, labels)
        elif self.config.validation_config.eval_metric == "bleu":
            return compute_bleu_score(predictions, labels)
        else:
            raise RuntimeError(f"No such metric: {self.config.validation_config.eval_metric}")



if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Run peft finetuning experiment")

    arg_parser.add_argument("-c", "--config", type=Path, action="append", help="Path to experiment config")

    RunConfig.register_cli(arg_parser)

    args = arg_parser.parse_args()

    config = RunConfig.from_file(args.config, args)

    runner = Runner(config)

    if not args.is_evaluate:
        runner.train()

    score = runner.evaluate()

    if runner.config.validation_config.eval_metric == "fscore":
        runner.write_run({
            "precision": score[0],
            "recall": score[1],
            "fscore": score[2],
            "support": score[3],
            "labels": score[4]
        })
        print(f"Precision: {score[0]} Recall: {score[1]} Fscore: {score[2]} Support: {score[3]} Labels: {score[4]}")
    else:
        runner.write_run(score)
        print(f"Score: {score}")
