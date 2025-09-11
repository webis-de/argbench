#!/usr/bin/env python3
import gc
import logging

import optuna
import re

from bert_score.utils import padding
from optuna import create_study
from optuna.samplers import TPESampler
from peft import (PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training, )
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import *
from vllm import LLM, SamplingParams
from vllm.distributed import destroy_model_parallel
from vllm.lora.request import LoRARequest

from argbench.experiment.hpo_output import HPOOutput
from argbench.experiment.leaderborad import Leaderboard
from argbench.experiment.memory_profiling import MemoryUsageCallback
from argbench.experiment.prepare_experiment import *
from argbench.experiment.segmentation_metric import *
from argbench.experiment.testing import *
from argbench.experiment.utils import *
logger = None

device = get_device()
cut_off_logged = False
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

def log_mem(message):
    if device =="cuda":
        for i in range(torch.cuda.device_count()):
            t = torch.cuda.mem_get_info(i)
            free_gpu, total_gpu = (t[0]/(1024**3),t[1]/(1024**3))
            used_cpu = (psutil.virtual_memory()[3]/1024**3)
            perc_memory = psutil.virtual_memory()[2]/100
            free_cpu_perc = 1 - perc_memory
            total_cpu = (1/perc_memory)*used_cpu
            free_cpu = total_cpu * free_cpu_perc
            logger.info(f"*** GPU Memory Device {i} {message}: {free_gpu:2.0f} GB free from {total_gpu:2.0f} GB  |  "
                        f" CPU Memory: {free_cpu:2.0f} GB free from {total_cpu:2.0f} GB")

def log_model_type(model):
    reported_modules = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        module_path = ".".join(name.split(".")[:-1]) if "." in name else ""

        if module_path not in reported_modules:
            logger.debug(f"Module: {module_path if module_path else 'Top-level'}, Representative Parameter: {name}, Precision (dtype): {param.dtype}")
            reported_modules.add(module_path)


def clean_prediction_general(prediction):
    if prediction.startswith("<|start_header_id|>assistant<|end_header_id|>"):
        prediction = prediction.replace("<|start_header_id|>assistant<|end_header_id|>", "")
    if prediction.startswith("assistant"):
        prediction = prediction.replace("assistant", "")
    prediction = prediction.replace("<s>", "")
    return prediction

def clean_for_generation_and_segmentation(prediction, chain_of_thoughts):
    prediction = clean_prediction_general(prediction)
    ## Output is the main indicator of output
    if "Output:" in prediction:
        index = prediction.rindex("Output:")
        prediction = prediction[index+7:]
    elif "output:" in prediction:
        index = prediction.rindex("output:")
        prediction = prediction[index+7:]
    ## Dorp thinking processes
    prediction = re.sub("<think>.*</think>", "", prediction)

    ## drop unjustified thinking tokens
    if "</think>" in prediction :
        prediction = prediction.replace("</think>", "")

    return prediction

def clean_for_classification(prediction, labels):
    prediction = prediction.lower().strip()
    prediction = clean_prediction_general(prediction)
    labels = "|".join([label.lower() for label in  labels])
    logger.debug(labels)
    regular_expressions = [ f"^({labels})",  f"({labels})$", f".*output:?\\s?({labels})", f".*</think>\\s?({labels})"]
    for regular_expression in regular_expressions:
        match  = re.match(regular_expression, prediction)
        if match:
            label = match.group(1)
            return label
    return prediction

### TODO replace by apply chat template
def formate_model_template(template):
    def formate_template(data_point):
        data_point["input"] = template.format(instruction=data_point["input"])
        return data_point
    return formate_template

def tokenize(prompt, tokenizer, cutoff_len, train):
    if train:
        #prompt = tokenizer(prompt, padding="max_length", max_length=cutoff_len, truncation=True)
        prompt = tokenizer(prompt, padding=True, max_length=cutoff_len, truncation=True)
        logger.debug( f"count of tokens in prompt {len(prompt['input_ids'])}")
    else:

        #prompt =  tokenizer(prompt, padding="max_length", max_length=cutoff_len, truncation=True, return_tensors="pt")
        prompt =  tokenizer(prompt, padding=True, max_length=cutoff_len, truncation=True, return_tensors="pt")

        logger.debug( f"count of tokens in prompt {len(prompt['input_ids'])}")
        return prompt
    if prompt["input_ids"][-1] != tokenizer.eos_token_id and len(prompt["input_ids"]) < cutoff_len:
        prompt["input_ids"].append(tokenizer.eos_token_id)
        prompt["attention_mask"].append(1)

    return prompt


def get_tokenizer(cutoff_len, tokenizer: AutoTokenizer, train: bool):


    def generate_and_tokenize_prompt(data_point):
        """
        Tokenizes data instance for feeding the model during training/testing

        :param data_point: Dict with "input", "output" strings
        :returns: tokenized prompt
        """
        input_prompt = tokenize(data_point['input'], tokenizer, cutoff_len, train)
        if train:
            full_prompt = tokenize(f"{data_point['input']}{data_point['output']}", tokenizer, cutoff_len, train)
            full_prompt["labels"] = full_prompt["input_ids"].copy()
            instruction_len = len(input_prompt) - 1
            full_prompt["labels"] = [-100] * instruction_len + full_prompt["labels"][instruction_len:]

            return full_prompt
        return input_prompt
    return generate_and_tokenize_prompt

def get_truncated_text(tokenizer):
    def generate_truncated(data_point):

        data_point["input"] = tokenizer.decode(data_point["input_ids"][0], skip_special_tokens=True)



        return data_point
    return generate_truncated

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
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.path, padding_side="left",
                                                       truncation=True, max_length = config.cutoff_len,
                                                       trust_remote_code=True, padding=False
                                                       )

        self.turn_off_thinking_qwen()
        self.tokenizer.pad_token = config.get_pad_token_id()
        self.tokenizer.unk_token = config.get_unk_token_id()
        self.tokenizer.add_special_tokens({'pad_token' : "<pad>"})
        self.generation_config = GenerationConfig(**config.generation_config.to_conf())
        self.task_metrics = get_evaluation_metrics_map()
        self.leaderboard = Leaderboard(config.get_leaderboard_path())
        if "name" in self.config.test_dataset:
            self.test_dataset_name = self.config.test_dataset["name"]
        else:
            if not self.config.prompting:
                raise ValueError("Error: Select Dataset")
            else:
                self.test_dataset_name = None


        self.load_data()
        self.prediction_samples = []

    def turn_off_thinking_qwen(self):
        if "qwen" in self.config.model and self.config.get_prompting_technique() != PromptingTechnique.COT:
            self.config.model_config.prompt_template += "\n\n<think>\n\n</think>\n\n"
            self.config.model_config.output_splitter = "</think>"

    def prepare_data(self):
        cutoff_len = self.config.cutoff_len
        template_formatter = formate_model_template(self.config.model_config.prompt_template)
        self.iterable_dataset = DatasetDict()
        if self.config.prompting:

            #tokenizer = get_tokenizer(cutoff_len, self.tokenizer, False)
            tokenized_template = self.tokenizer(self.config.model_config.prompt_template)
            tokenized_template_len = len(tokenized_template["input_ids"])

            logger.debug(f"length of tokenized template {tokenized_template_len}")

            tokenizer = get_tokenizer(cutoff_len - tokenized_template_len, self.tokenizer, False)
            generate_truncated = get_truncated_text(self.tokenizer)
            for task_label in self.dataset:
                task = task_label.replace("test_", "")
                log_mem(f"formatting {task}")
                self.iterable_dataset[task_label] = self.dataset[task_label].map(tokenizer).map(generate_truncated)
                log_mem(f"tokenizing {task}")
                self.iterable_dataset[task_label] = self.iterable_dataset[task_label].map(template_formatter)

        else:

            tokenizer = get_tokenizer(cutoff_len, self.tokenizer, True)
            for split in self.dataset:

                if split !="test":
                    self.iterable_dataset[split] = self.dataset[split].map(template_formatter).map(tokenizer)
                else:
                    tokenizer = get_tokenizer(cutoff_len, self.tokenizer, False)

                    if self.config.hpo:
                        log_mem(f"formatting {split} of {self.test_dataset_name}")
                        self.iterable_dataset["test"] = self.dataset["val"].map(template_formatter)
                    else:
                        self.iterable_dataset["test"] = self.dataset["test"].map(template_formatter)
                    log_mem(f"tokenizing {split} of {self.test_dataset_name}")
                    self.iterable_dataset["test"] = self.iterable_dataset["test"].map(tokenizer)
                    self.iterable_dataset["test"].set_format("pt", columns=["input_ids"], output_all_columns=True)



                # if split =="train":
                #     self.iterable_dataset[split] = self.dataset[split].to_iterable_dataset().map(template_formatter).map(tokenizer)
                # elif split == "val":
                #     self.iterable_dataset[split] = self.dataset[split].map(template_formatter).map(tokenizer)
                # else:
                #     tokenizer = get_tokenizer(cutoff_len, self.tokenizer, False)
                #     if self.config.hpo:
                #         log_mem(f"formatting {split} of {self.test_dataset_name}")
                #         self.iterable_dataset["test"] = self.dataset["val"].to_iterable_dataset().map(template_formatter)
                #     else:
                #         self.iterable_dataset["test"] = self.dataset["test"].to_iterable_dataset().map(template_formatter)
                #     log_mem(f"tokenizing {split} of {self.test_dataset_name}")
                #     self.iterable_dataset["test"] = self.iterable_dataset["test"].map(tokenizer)
                #

        log_mem(f"Finished preprocessing ")

    def prepare_model_for_training(self, quantization=False):
        """
        Prepare a model using configuration or HPO trial

        :param trial: Optuna trial object
        :param quantization: Bool

        :returns: Model to be trained
        """
        logger.info(f"preparing model {self.model_config.path} on {device}")

        model = self.prepare_model_for_causal_llm(self.model_config.path, quantization)



        logger.debug(f"logging model {self.config.model} precision")
        log_model_type(model)
        self.base_model = model



        if not self.config.prompting:
            if quantization:
                logger.info(f"preparing model for kbit training")
                model = prepare_model_for_kbit_training(model)
            model.enable_input_require_grads()
        logger.info("loaded model")
        if self.config.peft_configs and self.config.peft_fresh_config:
            raise RuntimeError("Cannot instantiate both fresh and trained models")
        if self.config.peft_configs:
            model = self.prepare_peft_model(model)
        if self.config.peft_fresh_config:
            model = self.prepare_new_peft_model(model)
        logger.debug(f"logging peft model {self.config.model} precision")
        log_model_type(model)

        
        self.peft_model = model
        log_mem(f"created model for training")
        return model

    def  prepare_model_for_generation(self):
        logger.info(f"running {self.model_config.path} on {device}")
        if self.config.peft_configs or self.config.peft_fresh_config:
            llm = LLM(model=self.model_config.path, enable_lora=True, seed=self.config.seed, device=device, trust_remote_code=True)
        else:
            if self.config.model == "llama-3.3-70b-instruct" :
                logger.info("running on 4 GPUS")
                llm = LLM(model=self.model_config.path, tensor_parallel_size=4, enable_lora=True, seed=self.config.seed, device=device, trust_remote_code=True)
            elif self.config.model == "deepseek-r1-distill-32b" or self.config.model == "qwen3-32b" or self.config.model == "gemma-3-27b-it" or self.config.model == "mixtral-8x7b":
                logger.info("running on 2 GPUS")
                llm = LLM(model=self.model_config.path, tensor_parallel_size=2, seed=self.config.seed, device=device, trust_remote_code=True)
            else:
                logger.info("running on 1 GPUS")
                llm = LLM(model=self.model_config.path, seed=self.config.seed, device=device, trust_remote_code=True)

            #llm = LLM(model=base_model)
        log_mem("after loading vllm model")
        return llm

    @staticmethod
    def get_generation_config_from_vllm_params(sampling_params: SamplingParams):
        temperature = sampling_params.temperature
        top_p = sampling_params.top_p
        top_k = sampling_params.top_k
        max_tokens = sampling_params.max_tokens
        min_tokens = sampling_params.min_tokens
        generation_config = GenerationConfig(max_new_tokens=max_tokens, min_new_tokens=min_tokens, top_p=top_p, top_k=top_k, temperature=temperature)
        return generation_config

    def load_sampling_params(self, test_dataset, trial=None, hpo_config=None):

        task_specific_vllm_config = None
        if self.config.chain_of_thoughts:
            task_generation_config = self.config.cot_task_generation_config
        else:
            task_generation_config = self.config.shot_task_generation_config

        for decoding_setup in task_generation_config:
            if decoding_setup in test_dataset:
                task_specific_vllm_config = task_generation_config[decoding_setup]
                logger.debug(f"using generation config for {test_dataset}")

        if not task_specific_vllm_config  and "default" in task_generation_config:
            logger.debug("using default generation config")
            task_specific_vllm_config = task_generation_config["default"]

        elif not task_specific_vllm_config :
            task_specific_vllm_config = self.config.vllm_config
            logger.debug(f"using central generation config")
        task_specific_vllm_config = task_specific_vllm_config.to_conf(trial, hpo_config)

        logger.debug(f"using {task_specific_vllm_config}")

        sampling_params = SamplingParams(**task_specific_vllm_config, truncate_prompt_tokens=self.config.cutoff_len)

        return sampling_params

    def prepare_trainer(self, model, trial=None, training_arg_hpo=None, data_collator_hpo=None, early_stopping_hpo=None):
        """
        Tokenizes train and test datasets and returns initialized Trainer instance

        :returns: Trainer initialized with configuration parameters from RunConfig object and tokenized data
        """
        if self.config.debug:
            report_to = "tensorboard"
            tensorboard_dir = self.config.get_tensorboard_log_dir()
        else:
            report_to = None
            tensorboard_dir = None
        training_args = self.config.training_args_config.to_conf(trial, training_arg_hpo)

        train_args = TrainingArguments(output_dir=self.config.get_output_path(),
            report_to=report_to,torch_empty_cache_steps=4,
            logging_dir=tensorboard_dir,dataloader_pin_memory=True,
            **training_args)

        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            **self.config.data_collator_config.to_conf(trial, data_collator_hpo,), max_length=self.config.cutoff_len
        )
        callbacks = []
        if self.config.debug:
            callbacks.append(MemoryUsageCallback(logger))

        if self.config.early_stopping_config:
            callbacks.append(
                EarlyStoppingCallback(**self.config.early_stopping_config.to_conf(trial, early_stopping_hpo))
            )
        log_mem("preparing trainer")

        trainer = Trainer(model=model, callbacks=callbacks, train_dataset=self.iterable_dataset["train"].with_format("torch"),
                          eval_dataset=self.iterable_dataset["val"].with_format("torch"),
        args=train_args, data_collator=data_collator)

        return trainer

    def load_data(self):

        experiment_type = self.config.get_experiment_type()
        prompting_technique = self.config.get_prompting_technique()
        sample = self.config.sample

        self.dataset = load_experiment(experiment_type,prompting_technique, sample, test_task= self.test_dataset_name, run_config=self.config, skill=self.config.skill_filter)

    def prepare_model_for_causal_llm(self, base_model, quantization):
        """
        Initializes ModelForCausalLM model and its quantization

        :param base_model: huggingface model path or name
        :param quantization: whether to train the model in a quantized manner
        :returns: ModelForCausalLM initialized from config
        """
        params = {"trust_remote_code":True, "pretrained_model_name_or_path":base_model}
        if quantization:
            params["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            params["device_map"]= "auto"
            return AutoModelForCausalLM.from_pretrained(**params)
        else:
            params["quantization_config"] = None
            #params["device_map"]= "auto"
            return AutoModelForCausalLM.from_pretrained(**params, torch_dtype=torch.bfloat16)

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

    def load_model(self):
        """Loads model checkpoint"""
        log_mem("loading model")
        model = self.prepare_model_for_training(self.config.quantization)

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

    def dump_predictions(self):
        if self.config.prediction_path:
            with open(self.config.prediction_path, "a+") as file:
                file.writelines(self.prediction_samples)

    def hpo_objective(self, trial: Trial):
        log_mem(f"loading model for hpo")
        self.model = self.load_model()
        #self.prepare_data()
        log_mem(f"loaded model for hpo")
        self.trainer = self.prepare_trainer(
            self.model,
            trial,
            self.config.hpo_config.training_args_config,
            self.config.hpo_config.data_collator_config,
            self.config.hpo_config.early_stopping_config
        )
        log_mem(f"started training")
        self.trainer.train()
        log_mem(f"trained model")

        self.free_model()


        sampling_params = self.load_sampling_params(self.test_dataset_name, trial, self.config.hpo_config.vllm_config)
        self.trainer.evaluate()
        metrics = self.evaluate(self.test_dataset_name, self.iterable_dataset["test"], sampling_params, model=self.trainer.model)
        log_mem(f"finished evaluation")
        logger.debug(f"metrics are {metrics}")
        metric = self.task_metrics[self.test_dataset_name]
        if "fscore" in metric:
            return metrics["fscore"]
        else:
            return metrics["generation-score"]

    def perform_hpo(self):
        """Perform HPO search"""

        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

        hpo_path = self.config.hpo_config.hpo_fine_grained_output + "/" + self.config.get_experiment_name() + ".log"
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(hpo_path),  # NFS path for distributed optimization
        )
        study = create_study(
            storage=self.config.hpo_config.storage,
            study_name=self.config.get_experiment_name(),
            direction=self.config.hpo_config.direction,
            sampler=TPESampler(),
            load_if_exists=True
        )
        study.optimize(self.hpo_objective, self.config.hpo_config.n_trials)
        return study.best_params, study.best_trial.value

    def execute(self):
        """pytho
        Execute training, hpo or evaluation
        """
        job_name = self.config.job_name
        if self.config.hpo:
            self.prepare_data()
            hpo_output = HPOOutput(self.config.hpo_config.hpo_coarse_output)
            now = datetime.now()
            self.starting_time = now.strftime("%m-%d-%H:%M:%S")
            best_params, best_value = self.perform_hpo()
            experiment_name = self.config.get_experiment_name()
            results = {"test_task": self.test_dataset_name, "metric" : self.config.hpo_config.val_metric, "score": best_value,
                       "experiment": experiment_name,  "model" : self.model_config.label  , "start_time": self.starting_time,
                       "learning_rate":best_params["learning_rate"], "batch_size":best_params["batch_size"], "job_name": job_name}

            hpo_output.add_results(results)

            logger.info(f"best params are {best_params}")
            logger.info(f"best value is {best_value}")

            hpo_output.save_file()
            return

        self.starting_time =datetime.now().strftime("%m-%d-%H:%M:%S")
        set_seed(self.config.seed)
        if not self.config.prompting:

            self.prepare_data()
            self.free_model()
            self.free_vllm_model()
            model = self.load_model()
            self.trainer = self.prepare_trainer(model)
            self.trainer.train()
            self.adapter_path = self.config.get_model_path(self.starting_time)
            self.trainer.save_model(self.adapter_path)
            for obj in self.trainer.state.log_history:
                logger.debug(obj)
            log_mem("trained")



        all_results = []
        if self.config.skill_filter:
            filter = config.skill_filter
        else:
            filter = "None"

        prompting_technique = ""
        if self.config.chain_of_thoughts:
            prompting_technique = "cot"
        if self.config.prompting:
            ## apply formatting function
            ## apply tokenziation function
            ## use data colloator without tokenization
            self.prepare_data()
            self.vllm = self.prepare_model_for_generation()
            for task_label in self.iterable_dataset:
                task = task_label.replace("test_", "")
                sampling_params= self.load_sampling_params(task)

                train_subsample_amount = self.config.train_datasets.get("subsample_amount", None)
                test_data =  self.iterable_dataset[task_label]
                metrics = self.evaluate(task, test_data, sampling_params, vllm=self.vllm)
                log_mem(f"tested on {task}")

                for metric in metrics:
                    results = {"test_task": task, "metric" : metric, "score": metrics[metric],  "model" : self.config.model + prompting_technique,
                               "start_time": self.starting_time, "k": train_subsample_amount, "filter": filter, "seed": self.config.seed, "job_name": job_name}
                    all_results.append(results)
                    self.leaderboard.add_results(results)
                log_mem(f"tested on {self.test_dataset_name}")

        else:
            ## apply formatting function
            ## apply tokenziation function
            ## use data colloator without tokenization
            sampling_params= self.load_sampling_params(self.test_dataset_name)
            train_subsample_amount = self.config.train_datasets.get("subsample_amount", None)
            self.trainer.evaluate()
            val_loss = self.trainer.state.log_history[-1]['val_loss']
            train_loss = self.trainer.state.log_history[-1]['train_loss']
            metrics = self.evaluate(self.test_dataset_name, self.iterable_dataset["test"], sampling_params, model=self.trainer.model)
            for metric in metrics:
                results = {"test_task": self.test_dataset_name, "metric" : metric, "score": metrics[metric],
                           "model" : self.config.model,  "start_time": self.starting_time, "k": train_subsample_amount,
                           "filter":filter, "seed": self.config.seed, "val_loss": val_loss, "train_loss": train_loss, "job_name":job_name}
                all_results.append(results)
                log_mem(f"tested on {self.test_dataset_name}")
                self.leaderboard.add_results(results)

            logger.debug(f" fine tuning metrics {metrics}")

        self.dump_predictions()
        return all_results

    @with_timing
    def evaluate(self, test_task_name, test_data, sampling_params, model=None, vllm=None):
        """
        Performs model evaluation using the test datasets and evaluation metric from ValidationConfig. The test
        dataset is the name of the test task and task_data is the test data points
        """
        #set_trace()

        log_mem(f"testing {test_task_name}")
        dataset = test_data
        labels = []
        predictions = []
        responses = []
        ## is the batch size here a bottleneck?
        if vllm:
            count_workers = 1
        else:
            count_workers = 8
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        prompting_technique = str(self.config.get_prompting_technique().value)

        ## If an adapter will be fine-tuned then an output dir is there
        if vllm:
            if self.config.peft_configs:
                logger.debug("++++ lora input ++++")
        elif model:
            generation_config = Runner.get_generation_config_from_vllm_params(sampling_params)
            model = torch.compile(model)
            model.eval()
        output_splitter = self.model_config.output_splitter
        counter = 0
        metric = self.task_metrics[test_task_name]
        if metric == "fscore":
            tmp_loader = DataLoader(dataset, batch_size=1)
            tmp_labels = [record["output"][0] for record in list(tmp_loader)]
            labels_lowered = list(set(tmp_labels))
            logger.debug(f"considered labels are {labels_lowered}")
        for data in tqdm(loader):
            text = data["input"][0]
            labels.extend(data["output"])

            if model:

                inputs = data["input_ids"][0].cuda()
                with torch.inference_mode():
                    generated = model.generate(input_ids=inputs, generation_config=generation_config, return_dict_in_generate=True)

                    output = self.tokenizer.batch_decode(generated.sequences)

                    output = [o.split(output_splitter)[-1] for o in output]

                    response = output[0]
                    responses.append(response)

                    if metric == "fscore":
                        prediction = clean_for_classification(response, labels_lowered)
                    else:
                        prediction = clean_for_generation_and_segmentation(response, self.config.chain_of_thoughts)
                    predictions.append(prediction)
                    if prediction:
                        logger.debug(format_logging(response, prediction, text))

            if vllm:

                if self.config.peft_configs and self.adapter_path:
                    lora_request = LoRARequest("sql_adapter", 1,self.adapter_path)
                    outputs = vllm.generate(text, sampling_params=sampling_params, lora_request=lora_request, use_tqdm=False)
                else:
                    outputs = vllm.generate(text, sampling_params=sampling_params, use_tqdm=False)
                for output in outputs:
                    response = output.outputs[0].text
                    responses.append(response)
                    counter +=1

                    if metric == "fscore":
                        prediction = clean_for_classification(response, labels_lowered)
                    else:
                        prediction = clean_for_generation_and_segmentation(response, chain_of_thoughts=self.config.chain_of_thoughts)
                    predictions += [prediction]
                    if prediction:
                        logger.debug(format_logging(response, prediction, text))
        if predictions:
            random_indices = [random.randint(0,len(predictions)-1) for _ in range(10)]
            sampled_predictions = [re.sub("\n+"," ",predictions[index]) for index in random_indices]
            sampled_responses = [re.sub("\n+"," ",responses[index]) for index in random_indices]
            sampled_labels = [re.sub("\n+"," ",labels[index]) for index in random_indices]
            sampled_predictions = zip(sampled_responses, sampled_predictions, sampled_labels, [self.config.model+"-"+prompting_technique for _ in range(10)], [test_task_name for _ in range(10)])
            sampled_predictions = [x[0]+"\t"+x[1]+"\t"+x[2]+"\t"+x[3]+"\t"+x[4]+"\t"+self.starting_time+"\t"+self.config.job_name+ "\n" for x in sampled_predictions]
            self.prediction_samples.extend(sampled_predictions)

        logger.debug(f"evaluating {counter} instances")

        if metric == "fscore":
            return compute_f1_score(predictions, labels)
        elif metric == "rouge":
            return compute_rouge_score(predictions, labels)
        elif metric == "generation-score":
            bleu = compute_bleu_score(predictions, labels)
            bert= compute_bert_score(predictions, labels)
            bleu.update(bert)
            average = (bleu["bleu"] + bert["bertscore"])/2
            bleu.update({"generation-score":average})
            return bleu
        elif metric == "meteor":
            return compute_meteor_score(predictions, labels)
        elif metric == "argument-fscore":
            return compute_seg_match_f1_score(predictions, labels,  ["argumentative", "non-argumentative"], ["non-argumentative"])
        elif metric == "aspect-fscore":
            return compute_seg_match_f1_score(predictions, labels,
                            ["aspect", "not-aspect"], ["not-aspect"])
        elif metric == "fallacy-fscore":
            return compute_seg_match_f1_score(predictions, labels,
                                              ["ad hominem", "appeal to emotion", "appeal to authority", "slippery slope", "false Cause", "slogans", "no-fallacy"], ["no-fallacy"])
        elif metric == "kendalltau":
            return compute_kendall_tau(predictions, labels)
        elif "dict" in metric:
            return {"fscore":0}
        else:
            raise RuntimeError(f"No such metric: {metric}")

if __name__ == "__main__":
    # try:
    #     #if os.path.exists("/mnt/home/yajjour"):
    #         #adjust_config("/bigwork/nhwpajjy","/mnt/home/yajjour")
    #     #if os.path.exists("/bigwork/nhwpajjy"):
    #         #adjust_config("/mnt/home/yajjour", "/bigwork/nhwpajjy")
    # except json.decoder.JSONDecodeError as error:
    #     print(error.msg)
    #     print (sys.exc_info())

    #turn_off_warnings()

    arg_parser = ArgumentParser(description="Run peft finetuning experiment")

    arg_parser.add_argument("-c", "--config", type=Path, action="append", help="Path to experiment config")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    RunConfig.register_cli(arg_parser)

    args = arg_parser.parse_args()
    config = RunConfig.from_file([], args)
    logger = get_logger(config)
    print(f"logging file is {config.log_path}")
    logger.info(" ".join(sys.argv[1:]))
    runner = Runner(config)


    score = runner.execute()


