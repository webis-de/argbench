#!/usr/bin/env python3
import os.path
import time

import pandas as pd
import psutil
import optuna
import sys
import gc

from argparse import ArgumentParser
from optuna import Trial, create_study
from torch.utils.data import DataLoader
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.distributed import destroy_model_parallel
from transformers import set_seed

from leaderborad import  Leaderboard
from datetime import datetime
from datasets import DatasetDict, Dataset
from hpo_output import HPOOutput
from utils import get_logger, get_evaluation_metrics_map
from prepare_experiment import collect_datasets
from testing import *
from tqdm import tqdm
from config import RunConfig

import json


logger = get_logger(__name__)

from filter_warnings import  *



def log_mem(message):
    t = torch.cuda.mem_get_info()
    free_gpu, total_gpu = (t[0]/(1024**3),t[1]/(1024**3))
    used_cpu = (psutil.virtual_memory()[3]/1024**3)
    perc_memory = psutil.virtual_memory()[2]/100
    free_cpu_perc = 1 - perc_memory
    total_cpu = (1/perc_memory)*used_cpu
    free_cpu = total_cpu * free_cpu_perc
    logger.info(f"*** GPU Memory {message}: {free_gpu:2.0f} GB free from {total_gpu:2.0f} GB  |  "
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



def with_timing(fn):
    def wrapper(*args, **kwargs):
        t = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            e= time.perf_counter()
            logger.debug(f"$$$ Time: for {fn} is {e-t:2.2f}")
            logger.debug(f"Time: for {fn} is {e-t:2.2f}")
    return wrapper

def eval_collate(batch):
    out_batch = {k: [] for k in batch[0]}

    for b in batch:
        for k in b:
            out_batch[k].append(b[k])

    return out_batch

class Runner:
    """Model runner class"""
    base_model = None
    peft_model = None
    vllm = None
    def __init__(self, config: RunConfig):
        """
        Initializes experiment runner with configuration object for training or evaluation

        :param config: RunConfig configuration object
        """
        self.config = config
        self.model_config = config.model_config
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.path, padding_side="left", unk_token="<unk>", truncation=True, max_length = config.data_collator_config.max_length)
        self.tokenizer.pad_token_id = config.pad_token_id


        self.generation_config = GenerationConfig(**config.generation_config.to_conf())
        self.task_metrics = get_evaluation_metrics_map()
        self.leaderboard = Leaderboard(config.leaderboard_path)
        if "name" in self.config.test_dataset:
            self.test_dataset_name = self.config.test_dataset["name"]
        if config.is_prompting:
            self.config.log_path = f"/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/logs/prompting-{self.config.base_model}.log"
        else:
            self.config.log_path = f"/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/logs/fine-tuning-{self.test_dataset_name}-{self.config.base_model}.log"

        self.prepare_data()

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
        logger.info("preparing model")

        model = self.prepare_model_for_causal_llm(
            self.model_config.path,
            self.config.quant_config.to_conf(trial, quant_hpo),
            self.config.model_config.to_conf(trial, llama_causal_hpo)
        )
        self.base_model = model

        if not self.config.is_prompting:
            model = prepare_model_for_kbit_training(model)
            model.enable_input_require_grads()
        logger.info("loaded model")
        if self.config.peft_configs and self.config.peft_fresh_config:
            raise RuntimeError("Cannot instantiate both fresh and trained models")
        if self.config.peft_configs:
            model = self.prepare_peft_model(model)
        if self.config.peft_fresh_config:
            model = self.prepare_new_peft_model(model)

        self.peft_model = model
        log_mem(f"created model for training")
        return model

    def prepare_model_for_generation(self):

        if self.config.peft_configs or self.config.peft_fresh_config:
            llm = LLM(model=self.model_config.path, enable_lora=True, seed=self.config.seed)
            #llm = LLM(model=base_model, enable_lora=True)
        else:
            llm = LLM(model=self.model_config.path, seed=self.config.seed)
            #llm = LLM(model=base_model)
        log_mem("after loading vllm model")
        return llm

    def load_sampling_params(self, test_dataset, trial=None, hpo_config=None):

        task_specific_vllm_config = None
        for decoding_setup in self.config.task_generation_config:
            if decoding_setup in test_dataset:
                task_specific_vllm_config = self.config.task_generation_config[decoding_setup]
                logger.debug(f"using generation config for {test_dataset}")

        if not task_specific_vllm_config  and "default" in self.config.task_generation_config:
            logger.debug("using default generation config")
            task_specific_vllm_config = self.config.task_generation_config["default"]

        elif not task_specific_vllm_config :
            task_specific_vllm_config = self.config.vllm_config
            logger.debug(f"using central generation config")
        task_specific_vllm_config = task_specific_vllm_config.to_conf(trial, hpo_config)

        logger.debug(f"using {task_specific_vllm_config}")

        sampling_params = SamplingParams(**task_specific_vllm_config)

        return sampling_params

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
        log_mem("preparing trainer")
        trainer = Trainer(
            model=model,
            callbacks=callbacks,
            train_dataset=self.train_data,
            eval_dataset=self.ft_test_data,
            args=train_args,
            data_collator=data_collator,
        )

        return trainer


    @with_timing
    def prepare_data(self):

        """
        Using configuration object collects train and test datasets
        """
        log_mem("preparing data")
        cutoff_len = self.config.cutoff_len
        train = True

        def generate_and_tokenize_prompt(data_point):
            """
            Tokenizes data instance for feeding the model during training/testing

            :param data_point: Dict with "input", "output" strings
            :returns: tokenized prompt
            """
            input_prompt = self.tokenize(data_point['input'], cutoff_len)
            if train:
                full_prompt = self.tokenize(f"{data_point['input']}{data_point['output']}", cutoff_len)
                instruction_len = len(input_prompt) - 1
                full_prompt["labels"] = [-100] * instruction_len + full_prompt["labels"][instruction_len:]
                return full_prompt
            return input_prompt

        self.prmt_test_data = {}

        if self.config.is_prompting:
            _, test_datasets = collect_datasets(self.config)
            for test_dataset in test_datasets:
                hf_test_dataset = Dataset.from_pandas(test_datasets[test_dataset])
                self.prmt_test_data[test_dataset] = hf_test_dataset.map(generate_and_tokenize_prompt, num_proc=12)
        else:
            train_datasets, test_dataset = collect_datasets(self.config)
            train_datasets = pd.concat(train_datasets.values(), axis=0).reset_index(drop=True)
            hf_train_dataset = Dataset.from_pandas(train_datasets)
            hf_test_dataset = Dataset.from_pandas(test_dataset)
            self.train_data = hf_train_dataset.map(generate_and_tokenize_prompt, num_proc=12)#, load_from_cache_file=f"/tmp/training_dataset.arrow")
            self.ft_test_data = hf_test_dataset.map(generate_and_tokenize_prompt, num_proc=12)
            logger.debug(f"counting {len(self.train_data)}")


        log_mem("prepared data")

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
            if config.peft_configs[0].model_id:
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

        model = get_peft_model(model, config, self.config.peft_fresh_config.adapter_name)
        for name, param in model.named_parameters():
            if 'lora' in name or 'Lora' in name:
                param.requires_grad = True
        return model


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
            full_prompt["labels"] = [-100] * instruction_len + full_prompt["labels"][instruction_len:]
            return full_prompt
        return input_prompt


    def load_model(self):
        """Loads model checkpoint"""
        log_mem("loading model")
        model = self.prepare_model_for_training()

        return model


    def free_model(self):
        if self.base_model:
            del self.base_model
        if self.peft_model:
            del self.peft_model
        torch.cuda.empty_cache()
        gc.collect()
        log_mem(f"saved and free model")


    def free_vllm_model(self):
        if self.vllm:
            destroy_model_parallel()
            del self.vllm
            gc.collect()
            torch.cuda.empty_cache()
            torch.distributed.destroy_process_group()


### This function is deprecated since it does not use VLLMs and is still dependent on one test dataset

    def hpo_objective(self, trial: Trial):

        self.model = self.prepare_model_for_training(
            trial,
            self.config.hpo_config.quant_config,
            self.config.hpo_config.model_config
        )

        self.trainer = self.prepare_trainer(
            self.model,
            trial,
            self.config.hpo_config.training_args_config,
            self.config.hpo_config.data_collator_config,
            self.config.hpo_config.early_stopping_config
        )

        self.trainer.train()
        log_mem(f"trained model")

#        self.trainer.save_model(self.config.training_args_config.output_dir + "/best-model")
        self.free_model()


        sampling_params = self.load_sampling_params(self.test_dataset_name, trial, self.config.hpo_config.vllm_config)
        self.trainer.evaluate()
        metrics = self.evaluate(self.test_dataset_name, self.ft_test_data, sampling_params, model=self.trainer.model)
        log_mem(f"finished evaluation")

        logger.debug(f"metrics are {metrics}")


        return metrics[self.config.hpo_config.val_metric]

    def perform_hpo(self):
        """Perform HPO search"""



        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

        hpo_path = self.config.hpo_config.hpo_fine_grained_output + "/" + self.config.hpo_config.study_name + ".log"
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(hpo_path),  # NFS path for distributed optimization
        )
        study = create_study(
            storage=storage,
            study_name=self.config.hpo_config.study_name,
            direction=self.config.hpo_config.direction,
            sampler=TPESampler(),
            load_if_exists=True
        )
        study.optimize(self.hpo_objective, self.config.hpo_config.n_trials)
        return study.best_params, study.best_trial.value

    def execute(self):
        """
        Execute training, hpo or evaluation
        """

        if self.config.is_hpo:
            # this should be changed to account for multiple datasets
            hpo_output = HPOOutput(self.config.hpo_config.hpo_coarse_output)
            now = datetime.now()
            starting_time = now.strftime("%m-%d-%H:%M:%S")
            best_params, best_value = self.perform_hpo()

            results = {"test_task": self.test_dataset_name, "metric" : self.config.hpo_config.val_metric, "score": best_value,
                       "experiment": "cross-task",  "model" : self.model_config.label  , "start_time": starting_time, "best-parameters":best_params}

            hpo_output.add_results(results)

            logger.info(f"best params are {best_params}")
            logger.info(f"best value is {best_value}")

            hpo_output.save_file()
            return

        starting_time =datetime.now().strftime("%m-%d-%H:%M:%S")
        set_seed(self.config.seed)
        if not self.config.is_prompting:
            self.free_model()
            self.free_vllm_model()
            model = self.load_model()
            self.trainer = self.prepare_trainer(model)
            self.trainer.train()
            self.trainer.save_model(self.config.training_args_config.output_dir + "/best-model")
            log_mem("trained")

        self.vllm = self.prepare_model_for_generation()


        all_results = []
        if self.config.is_prompting:
            for task in self.prmt_test_data:
                sampling_params= self.load_sampling_params(task)
                if "train_subsample_amount" in self.config.train_datasets:
                    train_subsample_amount = self.config.train_datasets["train_subsample_amount"]
                else:
                    train_subsample_amount = None
                test_data =  self.prmt_test_data[task]
                metrics = self.evaluate(task, test_data, sampling_params, vllm=self.vllm)
                log_mem(f"tested on {task}")
                for metric in metrics:
                    results = {"test_task": task, "metric" : metric, "score": metrics[metric],  "model" : self.config.base_model , "start_time": starting_time, "k": train_subsample_amount}
                    all_results.append(results)
                    self.leaderboard.add_results(results)
        else:
            sampling_params= self.load_sampling_params(self.test_dataset_name)
            if "train_subsample_amount" in self.config.train_datasets:
                train_subsample_amount = self.config.train_datasets["train_subsample_amount"]
            else:
                train_subsample_amount = None

            metrics = self.evaluate(self.test_dataset_name, self.ft_test_data, sampling_params, vllm=self.vllm)
            for metric in metrics:
                results = {"test_task": self.test_dataset_name, "metric" : metric, "score": metrics[metric],  "model" : self.config.base_model , "start_time": starting_time, "k": train_subsample_amount}
                all_results.append(results)
                log_mem(f"tested on {self.test_dataset_name}")
                self.leaderboard.add_results(results)

        self.leaderboard.save_file()

        return all_results


    def evaluate(self, test_task_name, test_data, sampling_params, model=None, vllm=None):
        """
        Performs model evaluation using the test datasets and evaluation metric from ValidationConfig. The test
        dataset is the name of the test task and task_data is the test data points
        """
        #set_trace()
        log_mem(f"testing {test_task_name}")
        dataset = test_data
        labels = test_data["output"]
        predictions = []

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=eval_collate,
            pin_memory=True
        )


        #trainer.model.eval()
        ## If an adapter will be fine-tuned then an output dir is there
        if vllm:
            if self.config.training_args_config.output_dir:
                adapter_path = self.config.training_args_config.output_dir + "/best-model"
                if os.path.exists(adapter_path):
                    lora_request = LoRARequest("adapter", 1, adapter_path+"/adapter")


        #set_trace()
        output_splitter = self.model_config.output_splitter
        for data in tqdm(loader):
            text = data["input"]
            if model:
                prompt = self.tokenizer(text=text, return_tensors="pt")
                inputs = prompt["input_ids"].cuda()

                generated = model.generate(
                    input_ids=inputs,
                    generation_config=self.generation_config,
                    return_dict_in_generate=True
                )

                output = self.tokenizer.batch_decode(generated.sequences)
                output = [o.split(output_splitter)[-1] for o in output]
                #set_trace()
                predictions.append(output[0])
            if vllm:
                set_trace()
                if self.config.peft_configs:
                    logger.debug("++++ lora input ++++")
                    outputs = vllm.generate(text, sampling_params=sampling_params, lora_request=lora_request, use_tqdm=False)
                else:
                    outputs = vllm.generate(text, sampling_params=sampling_params, use_tqdm=False)

                for output in outputs:

                    prediction = [output.outputs[0].text]
                    predictions += prediction
                    logger.debug(f"""got the
                                                       #################################
                                                        prediction:
                                                       ################################# 
                                                       f"{prediction}
                                                       ##################################
                                                       input
                                                       ##################################
                                                        {text}
                                                        #################################
                                                        """)

        metric = self.task_metrics[test_task_name]
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
        elif metric == "argument-bio-fscore":
            return compute_seg_bio_f1_score(predictions, labels, test_data["document"])
        elif metric == "aspect-bio-fscore":
            return compute_aspect_bio_f1_score(predictions, labels, test_data["document"])
        elif metric == "fallacy-bio-fscore":
            return compute_fallacy_bio_f1_score(predictions, labels, test_data["document"])
        elif metric == "sentence-fscore":
            return compute_sentence_f1(predictions, labels, test_data["document"])
        elif metric == "kendalltau":
            return compute_kendall_tau(predictions, labels)
        else:
            raise RuntimeError(f"No such metric: {self.config.validation_config.eval_metric}")

if __name__ == "__main__":


    #turn_off_warnings()
    arg_parser = ArgumentParser(description="Run peft finetuning experiment")

    arg_parser.add_argument("-c", "--config", type=Path, action="append", help="Path to experiment config")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    RunConfig.register_cli(arg_parser)

    args = arg_parser.parse_args()
    config = RunConfig.from_file([], args)

    print(f"logging file is {config.log_path}")
    runner = Runner(config)
    score = runner.execute()


