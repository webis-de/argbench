#!/usr/bin/env python3
import os.path
import logging
import psutil
from argparse import ArgumentParser
from optuna import Trial, create_study
from torch.utils.data import DataLoader
from pathlib import Path
from vllm import LLM, SamplingParams
from vllm.  lora.request import LoRARequest
from IPython.core.debugger import set_trace
import torch.autograd.profiler as profiler
from leaderborad import  Leaderboard

logger = logging.getLogger(__name__)

from dataclasses import asdict
import gc

import transformers
from filter_warnings import  *



logging.basicConfig(level=logging.INFO, format='%(message)s')

def log_mem(message):
    t = torch.cuda.mem_get_info()
    free_gpu, total_gpu = (t[0]/(1024**3),t[1]/(1024**3))
    used_cpu = (psutil.virtual_memory()[3]/1024**3)
    perc_memory = psutil.virtual_memory()[2]/100
    free_cpu_perc = 1 - perc_memory
    total_cpu = (1/perc_memory)*used_cpu
    free_cpu = total_cpu * free_cpu_perc
    logger.log(level=logging.INFO,msg=f"GPU Memory {message}: {free_gpu:2.0f} GB free from {total_gpu:2.0f} GB  |  "
                                         f" CPU Memory: {free_cpu:2.0f} GB free from {total_cpu:2.0f} GB")


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
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


logger = logging.getLogger(__name__)



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
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model, padding_side="left", unk_token="<unk>", truncation=True, max_length = config.data_collator_config.max_length)
        self.tokenizer.pad_token_id = config.pad_token_id

        log_mem("before preparing data")
        self.prepare_data()
        log_mem("after preparing data")

        logger.log(level=logging.INFO, msg="Data prepared!")
        logger.log(level=logging.INFO, msg=f"counting {len(self.train_data)}")
        self.generation_config = GenerationConfig(**config.generation_config.to_conf())
        self.task_metrics = json.load(open(config.task_metrics_path))
        self.leaderborad = Leaderboard(config.leaderboard_path)

    def prepare_model_for_training(self,
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
        logger.log(level=logging.INFO,msg="Prepare model")

        model = self.prepare_model_for_causal_llm(
            self.config.base_model,
            self.config.quant_config.to_conf(trial, quant_hpo),
            self.config.model_config.to_conf(trial, llama_causal_hpo)
        )
        self.base_model = model

        if not self.config.is_eval:
            model = prepare_model_for_kbit_training(model)
            model.enable_input_require_grads()
        logger.log(level=logging.INFO,msg="Model loaded")
        if self.config.peft_configs and self.config.peft_fresh_config:
            raise RuntimeError("Cannot instantiate both fresh and trained models")
        if self.config.peft_configs:
            model = self.prepare_peft_model(model)
        if self.config.peft_fresh_config:
            model = self.prepare_new_peft_model(model)

        self.peft_model = model
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
            args=train_args,
            data_collator=data_collator,
        )

        return trainer

    def prepare_data(self):
        """
        Using configuration object collects train and test datasets
        """
        self.train_datsets, self.test_datsets = collect_datasets(
            self.config
        )

        self.train_data = []
        for train_dataset in self.train_datsets:
            train_df = self.train_datsets[train_dataset].df
            for row in train_df.iterrows():
                row = row[1]
                processed = self.generate_and_tokenize_prompt(row, self.config.cutoff_len)
                self.train_data.append(processed)

        self.test_data = {}
        for test_dataset in self.test_datsets:
            test_df = self.test_datsets[test_dataset].df
            self.test_data[test_dataset] = []
            for row in test_df.iterrows():
                row = row[1]
                processed = self.generate_and_tokenize_prompt(row, self.config.cutoff_len, False)
                self.test_data[test_dataset].append(processed)


    def prepare_model_for_causal_llm(self, base_model, quant_config, model_config):
        """
        Initializes ModelForCausalLM model and its quantization

        :param base_model: huggingface model path or name
        :param quant_config: Quantization config
        :param model_config: Configuration parameters for model
        :returns: ModelForCausalLM initialized from config
        """
        quant_conf = BitsAndBytesConfig(**quant_config)
        return AutoModelForCausalLM.from_pretrained(
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

        model = self.prepare_model_for_training()
        return model


    def free_model(self):
        del self.base_model
        del self.peft_model
        torch.cuda.empty_cache()
        gc.collect()


    def hpo_objective(self, trial: Trial):

        model = self.prepare_model(
            trial,
            self.config.hpo_config.quant_config,
            self.config.hpo_config.model_config
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



        if not self.config.is_eval:
            log_mem("before loading model")

            model = self.load_model()

            log_mem("after loading model and before training")

            self.trainer = self.prepare_trainer(model)

            self.trainer.train()
            log_mem("after training")

            self.trainer.save_model(self.config.training_args_config.output_dir + "/best-model")

        base_model = self.config.base_model
        log_mem("before loading vllm model")
        if self.config.peft_configs:
            llm = LLM(model=base_model, enable_lora=True, tokenizer_mode="slow")
        else:
            llm = LLM(model=base_model)
        log_mem("after loading vllm model")
        all_results = []
        for test_dataset in self.test_datsets:
            task_df =  self.test_datsets[test_dataset].df
            log_mem(f"before testing on {test_dataset}")

            metrics = self.evaluate(llm, test_dataset, task_df)

            log_mem(f"after testing on {test_dataset}")
            for metric in metrics:
                results = {"test_task": test_dataset, "metric" : metric, "score": metrics[metric], "training_data": "MOC",  "model" : self.config.base_model }
                all_results.append(results)
                self.leaderborad.add_results(results)
            self.leaderborad.save_file()
        if self.base_model:
            self.free_model()
        return all_results


    def write_run(self, run_results):
        """
        Writes run results of training
        """
        with open(self.config.run_output_path, "w") as f:
            run_data = {
                "run_sesults": run_results
            }
            json.dump(run_data, f)


    def evaluate(self, llm,  test_dataset, task_data):
        """
        Performs model evaluation using the test datasets and evaluation metric from ValidationConfig. The test
        dataset is the name of the test task and task_data is the test data points
        """
        #set_trace()

        dataset = PandasDataset(task_data)
        labels = task_data["output"].tolist()
        predictions = []

        loader = DataLoader(
            dataset,
            batch_size=self.config.validation_config.batch_size,
            shuffle=True,
            collate_fn=eval_collate,
            pin_memory=True
        )


        #trainer.model.eval()

        adapter_path = self.config.training_args_config.output_dir + "/best-model"

        if test_dataset in self.config.task_generation_config:
            logger.log(level=logging.INFO, msg=f"using generation config for {test_dataset}")
            task_specific_vllm_config = self.config.task_generation_config[test_dataset]
        elif "default" in self.config.task_generation_config:
            logger.log(level=logging.INFO, msg=f"using default generation config")
            task_specific_vllm_config = self.config.task_generation_config["default"]
        else:
            task_specific_vllm_config = self.config.vllm_config.to_conf()
            logger.log(level=logging.INFO, msg=f"using central generation config")

        if os.path.exists(adapter_path):
            lora_request = LoRARequest("adapter", 1, adapter_path+"/adapter")

        sampling_params = SamplingParams(**task_specific_vllm_config.to_conf())

        #set_trace()
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
            if self.config.peft_configs:
                outputs = llm.generate(text, sampling_params=sampling_params, lora_request=lora_request)
            else:
                outputs = llm.generate(text, sampling_params=sampling_params)

            for output in outputs:
                #output = [o[len(text[i]):] for i, o in enumerate(output)]
                prediction = [output.outputs[0].text]
                predictions += prediction
                logger.log(level=logging.INFO, msg=f"got the prediction {prediction} for input{text}")
        #trainer.model.train()

        metric = self.task_metrics[test_dataset]

        if metric == "fscore-detailed":
            return compute_precision_recall_fscore_support(
                predictions,
                labels,
                f1_average=self.config.validation_config.fscore_average,
                beta=self.config.validation_config.fscore_beta
            )
        elif metric == "fscore":
            return compute_f1_score(predictions, labels)
        elif metric == "rouge":
            return compute_rouge_score(predictions, labels)
        elif metric == "bleu":
            return compute_bleu_score(predictions, labels)
        elif metric == "meteor":
            return compute_meteor_score(predictions, labels)
        elif metric == "bio-fscore":
            return compute_bio_f1_score(predictions, labels, task_data["document"].tolist())
        elif metric == "sentence-fscore":
            return compute_sentence_f1(predictions, labels, task_data["document"].tolist())
        elif metric == "kendalltau":
            return compute_kendall_tau(predictions, labels)
        else:
            raise RuntimeError(f"No such metric: {self.config.validation_config.eval_metric}")




if __name__ == "__main__":
    turn_off_warnings()
    arg_parser = ArgumentParser(description="Run peft finetuning experiment")

    arg_parser.add_argument("-c", "--config", type=Path, action="append", help="Path to experiment config")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    RunConfig.register_cli(arg_parser)

    args = arg_parser.parse_args()
    config_list = ["/home/yamen/projects/task-specific-argument-mining-and-generation/src/experiment/configs/complete_leave_one_out_ajjour17.json"]
    config = RunConfig.from_file(config_list)
    runner = Runner(config)
    score = runner.execute()


