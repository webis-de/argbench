#!/usr/bin/env python3
import os.path
from argparse import ArgumentParser
from optuna import Trial, create_study
from torch.utils.data import DataLoader
from pathlib import Path
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from dataclasses import asdict

import torch.autograd.profiler as profiler


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
from optuna.samplers import TPESampler
from peft import (
    PeftModel,
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)
from preprocess import collect_datasets, PandasDataset
from testing import *
from tqdm import tqdm
from config import RunConfig
import numpy as np
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
        """
        Initializes experiment runner with configuration object for training or evaluation

        :param config: RunConfig configuration object
        """
        self.config = config
        self.tokenizer = LlamaTokenizerFast.from_pretrained(config.base_model, padding_side="left", unk_token="<unk>", truncation=True, max_length = config.data_collator_config.max_length)
        self.tokenizer.pad_token_id = config.pad_token_id
        with profiler.record_function("preparing data"):
            self.prepare_data()
        print("Data prepared!")

        self.generation_config = GenerationConfig(**config.generation_config.to_conf())


    def prepare_model(self,
                      trial=None,
                      quant_hpo=None,
                      llama_causal_hpo=None):
        """
        Prepare a model using configuration or HPO trial

        :param trial: Optuna trial object
        :param quant_hpo: Quantization hyperparameters
        :param llama_causal_hpo: Model hyperparameters
        :param peft_hpo: PEFT model hyperparameters
        :param new_peft_hpo: New peft hyperparameters
        :returns: Model to be trained
        """
        print("Prepare model")
        model = self.prepare_llama_for_causal_llm(
            self.config.base_model,
            self.config.quant_config.to_conf(trial, quant_hpo),
            self.config.llama_causal_config.to_conf(trial, llama_causal_hpo)
        )
        if not self.config.is_eval:
            model = prepare_model_for_kbit_training(model)
            model.enable_input_require_grads()
        print("Model loaded")
        if self.config.peft_configs and self.config.peft_fresh_config:
            raise RuntimeError("Cannot instantiate both fresh and trained models")
        if self.config.peft_configs:
            model = self.prepare_peft_model(model)
        if self.config.peft_fresh_config:
            model = self.prepare_new_peft_model(model)

        return model


    def prepare_trainer(self,
                        model,
                        trial=None,
                        training_arg_hpo=None,
                        data_collator_hpo=None,
                        early_stopping_hpo=None):
        """
        Tokenizes train and test datasets and returns initialized Trainer instance

        :returns: Trainer initialized with configuration parameters from RunConfig object and tokenized data
        """
        train_args = TrainingArguments(
            **self.config.training_args_config.to_conf(trial, training_arg_hpo)
        )

        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            **self.config.data_collator_config.to_conf(trial, data_collator_hpo)
        )

        callbacks = []

        if self.config.early_stopping_config:
            callbacks = [
                EarlyStoppingCallback(**self.config.early_stopping_config.to_conf(trial, early_stopping_hpo))
            ]

        trainer = Trainer(
            model=model,
            callbacks=callbacks,
            train_dataset=self.train_data,
            eval_dataset=self.test_data,
            args=train_args,
            data_collator=data_collator,
        )

        return trainer

    def prepare_data(self):
        """
        Using configuration object collects train and test datasets
        """
        self.train_instances, self.test_instances = collect_datasets(
            self.config
        )

        self.train_data = []
        for row in self.train_instances.iterrows():
            row = row[1]
            processed = self.generate_and_tokenize_prompt(row, self.config.cutoff_len)
            self.train_data.append(processed)

        self.test_data = []
        for row in self.test_instances.iterrows():
            row = row[1]
            processed = self.generate_and_tokenize_prompt(row, self.config.cutoff_len, False)
            self.test_data.append(processed)


    def prepare_llama_for_causal_llm(self, base_model, quant_config, model_config):
        """
        Initializes LlamaForCausalLM model and its quantization

        :param base_model: huggingface model path or name
        :param quant_config: Quantization config
        :param model_config: Configuration parameters for model
        :returns: LlamaForCausalLM initialized from config
        """
        quant_conf = BitsAndBytesConfig(**quant_config)
        return LlamaForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            quantization_config=quant_conf,
            **model_config
        )


    def prepare_peft_model(self, model):
        """
        loads one or many combined peft models

        If RunConfig contains many PeftConfigs, then multiple peft adapeters are combined together

        :param model: Base LLM model
        :returns: PeftModel prepared for original model
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

        # model = get_peft_model(model, **self.config.peft_configs[0].to_conf())
        model = PeftModel.from_pretrained(
            model,
            **self.config.peft_configs[0].to_conf()
        )
        return model


    def prepare_new_peft_model(self, model):
        """
        Initializes new Peft adapter instead of loading ready one.

        :param model: Base LLM model
        :returns: PeftModel prepared for original model
        """
        if self.config.peft_fresh_config.adapter_type == "lora":
            config = LoraConfig(**self.config.peft_fresh_config.config_args)
        else:
            raise RuntimeError(f"No such adapter type: {self.config.peft_fresh_config.adapter_type}")

        return get_peft_model(model, config, self.config.peft_fresh_config.adapter_name)


    def tokenize(self, prompt, cutoff_len, add_eos_token=True):
        """
        Tokenize instance for training or testing

        :param prompt: Input prompt string
        :param cutoff_len: Max length of prompt in tokens
        :param add_eos_token: Should end of string token be added
        :returns: Dict with tokenization result
        """
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

        return result


    def generate_and_tokenize_prompt(self, data_point, cutoff_len, train=True):
        """
        Tokenizes data instance for feeding the model during training/testing

        :param data_point: Dict with "input", "output" strings
        :param cutoff_len: Prompt max length
        :param train: Is instance for training
        :returns: tokenized prompt
        """
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


    def load_model(self):
        """Loads model checkpoint"""
        model = self.prepare_model()
        return self.prepare_trainer(model)


    def hpo_objective(self, trial: Trial):

        model = self.prepare_model(
            trial,
            self.config.hpo_config.quant_config,
            self.config.hpo_config.llama_causal_config
        )

        trainer = self.prepare_trainer(
            model,
            trial,
            self.config.hpo_config.training_args_config,
            self.config.hpo_config.data_collator_config,
            self.config.hpo_config.early_stopping_config
        )

        trainer.train()

        test_result = self.evaluate(trainer)
        if self.config.hpo_config.val_metric:
            return test_result[self.config.hpo_config.val_metric]
        return test_result


    def perform_hpo(self):
        """Perform HPO search"""

        study = create_study(
            storage=self.config.hpo_config.storage,
            study_name=self.config.hpo_config.study_name,
            direction=self.config.hpo_config.direction,
            sampler=TPESampler(),
            load_if_exists=True
        )

        study.optimize(self.hpo_objective, self.config.hpo_config.n_trials)


    def execute(self):
        """
        Execute training, hpo or evaluation
        """
        if self.config.is_hpo:
            self.perform_hpo()
            return
        with profiler.record_function("loading model"):
            self.trainer = self.load_model()

        if not self.config.is_eval:
            with profiler.record_function("loading model"):
                self.trainer.train()

        return self.evaluate(self.trainer)


    def write_run(self, run_results):
        """
        Writes run results of training
        """
        with open(self.config.run_output_path, "w") as f:
            run_data = {
                "run_sesults": run_results
            }
            json.dump(run_data, f)


    def evaluate(self, trainer):
        """
        Performs model evaluation using test set and evaluation metric from ValidationConfig
        """
        labels = self.test_instances["output"]
        dataset = PandasDataset(self.test_instances)
        predictions = []

        loader = DataLoader(
            dataset,
            batch_size=self.config.validation_config.batch_size,
            shuffle=True,
            collate_fn=eval_collate,
            pin_memory=True
        )
        base_model = self.config.base_model

        #trainer.model.eval()

        adapter_path = self.config.training_args_config.output_dir
        llm = LLM(model=base_model, enable_lora=True, tokenizer_mode="slow")
        if os.path.exists(adapter_path):
            lora_request = LoRARequest("adapter", 1, adapter_path)
            sampling_params = SamplingParams(temperature=0, top_p=0, lora_request=lora_request)
        else:
            raise ValueError("no model is trained !")

        for data in tqdm(loader):
            text = data["input"]
            #prompt = self.tokenizer(text=text, return_tensors="pt", padding=True)
            #inputs = prompt["input_ids"].cuda()

            #generated = trainer.model.generate(
            #    input_ids=inputs,
            #    generation_config=self.generation_config,
            #    return_dict_in_generate=True
            #)
            #output = self.tokenizer.batch_decode(generated.sequences, skip_special_tokens=True)
            # output = self.tokenizer.decode(gen_diff[0])
        outputs = llm.generate(data["input"].values, sampling_params=sampling_params)
        for output in outputs:
            output = [o[len(text[i]):] for i, o in enumerate(output)]
            predictions += output

        #trainer.model.train()

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
        elif config.validation_config.eval_metric == "meteor":
            return compute_meteor_score(predictions, labels)
        elif config.validation_config.eval_metric == "bio-fscore":
            return compute_bio_f1_score(predictions, labels, dataset["input"].values())
        elif config.validation_config.eval_metric == "sentence-fscore":
            return compute_sentence_f1(predictions, labels, dataset["input"].values())
        else:
            raise RuntimeError(f"No such metric: {self.config.validation_config.eval_metric}")



if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Run peft finetuning experiment")

    arg_parser.add_argument("-c", "--config", type=Path, action="append", help="Path to experiment config")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    RunConfig.register_cli(arg_parser)

    args = arg_parser.parse_args()

    config = RunConfig.from_file(args.config, args)

    task_metrics = json.loads(open("configs/config_task_metrics_complete.json"))
    with profiler.profile(with_stack=True, profile_memory=True) as prof:
        runner = Runner(config)
        score = runner.execute()
        prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5)

    if runner.config.validation_config.eval_metric == "fscore":
        runner.write_run({
            "precision": score[0],
            "recall": score[1],
            "fscore": score[2],
            "support": score[3],
            "labels": score[4]
        })
        print(f"Precision: {score[0]} Recall: {score[1]} Fscore: {score[2]} Support: {score[3]} Labels: {score[4]}")
    elif runner.config.validation_config.eval_metric == "sentence-fscore":
        runner.write_run({"fscore": score["fscore"],
                          })
    elif runner.config.validation_config.eval_metric == "bio-fscore":
        runner.write_run({"fscore": score["fscore"],
                          "argb-fscore" : score["agb-fscore"],
                          "argi-fscore" : score["agi-fscore"],
                          "argo-fscore" : score["ago-fscore"],
                          })
    else:
        runner.write_run(score)
        print(f"Score: {score}")
