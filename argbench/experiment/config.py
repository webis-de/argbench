import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List
from datetime import datetime
from optuna import Trial
from peft.config import PeftConfig
from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING
#from utils import  rewrite_config
class ExperimentType(Enum):
    IN_TASK = "in-task"
    LEAVE_ONE_TASK = "leave-one-task"
    PROMPTING = "prompting"
    SKILL_TRANSFER = "skill-transfer"

class DatasetSplit(Enum):
    TRAIN = "train"
    TEST = "test"
    VAL = "val"
    TRAIN_AND_VAL = "train-and-val"

class PromptingTechnique(Enum):
    COT = "chain-of-thought"
    ZERO_SHOT = "zero-shot"
    ONE_SHOT = "one-shot"
    FOUR_SHOT = "four-shot"

def rewrite_config(config: dict, root_path:Path, new_root_path: Path):
    """
    Rewrite the paths in a dictionary so that you substitute /bigwork/nhwpajjy with the new root path
    :param config: the dictionary containing the configuration
    :param new_root_path: the new root path path
    :return:
    """

    for key in config:
        if isinstance(config[key],str):
            config[key] = config[key].replace(root_path, new_root_path)
        elif isinstance(config[key],dict):
            rewrite_config(config[key], root_path, new_root_path)
        elif isinstance(config[key],list):
            for obj in config[key]:
                if isinstance(obj, dict):
                    rewrite_config(obj, root_path, new_root_path)
        else:
            pass
    return config


def update_conf(config_update, config_other):
    for k, v in config_other.items():
        if isinstance(v, dict):
            config_update[k] = update_conf(config_update.get(k, {}), v)
        else:
            config_update[k] = v
    return config_update

def hpo_to_suggestion(trial: Trial, hpo: dict):
    """
    Turn hyperparameter configuration to hpo suggestion

    :param trial: Optuna Trial object
    :param hpo: hyperparameter config dict
    :returns: HP value
    """
    if hpo["type"] == "categorical":
        return trial.suggest_categorical(hpo["name"], hpo["choices"])
    if hpo["type"] == "int":
        return trial.suggest_int(hpo["name"], hpo["low"], hpo["high"], step=hpo.get("step", 1), log=hpo.get("log", False))
    if hpo["type"] == "float":
        return trial.suggest_float(hpo["name"], hpo["low"], hpo["high"], step=hpo.get("step", None), log=hpo.get("log", False))

class CommonConfig:
    """Methods for all config classes"""

    def to_conf(self, trial=None, suggested_hps=None):
        """
        Convert configuration object to dictionary
        """
        config_params = {k: v for k, v in self.__dict__.items() if v is not None}
        if not suggested_hps or not trial:
            return config_params
        for hp in suggested_hps:
            config_params[hp] = hpo_to_suggestion(trial, suggested_hps[hp])
        return config_params

@dataclass
class DataCollatorConfig(CommonConfig):
    """Configuration args for DataCollatorForSeq2Seq"""

    padding: bool = True

    return_tensors: str = "pt"

    max_length: int = None

    pad_to_multiple_of: int = None

    label_pad_token_id: int = None

@dataclass
class TrainingArgsConfig(CommonConfig):
    """TrainingArguments configuration arguments"""

    per_device_train_batch_size: int

    num_train_epochs: int

    learning_rate: float

    optim: str

    eval_strategy: str

    save_strategy: str

    eval_steps: int

    save_steps: int

    save_total_limit: int = 3

    weight_decay: float = 0.0

    per_device_eval_batch_size: int = 1

    adam_beta1: float = 0.9

    adam_beta2: float = 0.999

    adam_epsilon: float = 1e-8

    max_grad_norm: float = 1.0

    warmup_steps: int = None

    fp16: bool = False

    logging_steps: int = 10

    lr_scheduler_type: str = "constant_with_warmup"

    metric_for_best_model: str = None

    tf32: bool = False

    bf16: bool = False

    gradient_checkpointing: bool = True

    load_best_model_at_end: bool = True

    group_by_length: bool = False

    do_eval: bool = True




@dataclass
class LLamaCausalConfig(CommonConfig):
    """LlamaForCausalLM configuration arguments"""

    device_map: str = "auto"



@dataclass
class PeftPretrainedConfig(CommonConfig):
    """PEFT configuration arguments"""

    model_id: str

    adapter_name: str

    adapter_type: str = None

    is_trainable: bool = True

    adapter_weight: float = None

    config: PeftConfig = None

    def to_conf(self, trial=None, suggested_hps=None):
        config = super().to_conf(trial, suggested_hps)
        if config.get("adapter_weight"):
            del config["adapter_weight"]
        if config.get("adapter_type"):
            del config["adapter_type"]
        return config

@dataclass
class PeftAdapterConfig(CommonConfig):
    """Configuration object for peft adapter config"""

    # Type of adapter to instanciate
    # loha
    # llama-adapter
    # boft
    adapter_type: str

    adapter_name: str

    config_args: dict

    def to_conf(self, trial=None, suggested_hps=None):
        config = super().to_conf(trial, suggested_hps)
        del config["adapter_type"]
        return config

@dataclass
class ValidationConfig(CommonConfig):
    """Configuration object for model evaluation"""

    # Evaluation metric name
    # fscore
    # rouge
    # bleu
    eval_metric: str

    batch_size: int = 16

    fscore_average: str = None

    fscore_beta: int = 1.0



@dataclass
class ModelGenerationConfig(CommonConfig):
    """Configuration object for GenerationConfig"""

    max_length: int = 20

    max_new_tokens: int = None

    min_length: int = 0

    min_new_tokens: int = None

    # Early stopping condition
    # True
    # False
    # "never"
    early_stopping: str = None

    max_time: float = None

    # Generation strategy parameters

    do_sample: bool = False

    num_beams: int = 1

    num_beam_groups: int = 1

    penalty_alpha: float = None

    use_cache: bool = True

    # Logit parameters

    temperature: float = 1

    top_k: int = 50

    top_p: float = 1

    min_p: float = None

    typical_p: float = None

    epsilon_cutoff: float = None

    eta_cutoff: float = None

    diversity_penalty: float = 0

    repetition_penalty: float = 1

    length_penalty: float = 1

    def to_conf(self):
        config = super().to_conf()

        return config


@dataclass
class EarlyStoppingConfig(CommonConfig):
    """Config for early stopping"""

    early_stopping_patience: int = 1

    early_stopping_threshold: float = 0.0

@dataclass
class HPOConfig(CommonConfig):
    """Config for HPO"""

    n_trials: int

    storage: str

    study_name: str

    direction: str
    hpo_fine_grained_output: str
    hpo_coarse_output: str
    val_metric: str = None

    model_config: dict = field(default_factory=dict)

    # Training configs
    training_args_config: dict = field(default_factory=dict)
    early_stopping_config: dict = field(default_factory=dict)
    data_collator_config: dict = field(default_factory=dict)
    vllm_config: dict = field(default_factory=dict)
    generation_config: dict = field(default_factory=dict)


@dataclass
class ModelConfig(CommonConfig):
    label: str
    path: str
    prompt_template: str

    output_splitter: str

    cutoff_len: int

@dataclass
class VLLMGenerationConfig(CommonConfig):
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    min_tokens: int

@dataclass
class RunConfig:
    """Config for instruction finetuning run"""

    test_dataset: dict


    # Dataset metrics a dictionary that contains for each task which metric will be used
    task_metrics_path: str

    output_dir: str
    cutoff_len: int
    # Model Task-specific configuration config path
    best_hyper_parameters_path: str
    generation_config_path: str

    debug: bool


    # task-specific

    # Seed to use
    seed: int
    # Prompt cutoff length

    # Training datasets
    train_datasets: dict
    # Test datasets

    model_configs: List[ModelConfig]

    experiment_splits_path: str


    models_folder: str


    # Data folder
    data_folder: str
    # Base model path
    base_model: str
    # Should only evaluation be performed
    prompting: bool
    # Should HPO be performed

    tensorboard_logs: str

    argbench_dataset_path: str

    hpo: bool
    # Padding token id
    chain_of_thoughts : bool = False


    prediction_path: str = None
    in_task: bool = False
    sample: bool = False

    skill_filter: str = None

    model_config: ModelConfig = None
    pad_token_id: int = 0
    # Peft combination type
    combination_type: str = None
    # Data type
    data_type: str = "ndjson"

    log_path: str = None



    # Peft finetuning configs
    peft_configs: List[PeftPretrainedConfig] = None
    peft_fresh_config: PeftAdapterConfig = None
    # Training configs
    training_args_config: TrainingArgsConfig = None
    early_stopping_config: EarlyStoppingConfig = None

    data_collator_config: DataCollatorConfig = None
    generation_config: ModelGenerationConfig = None
    validation_config: ValidationConfig = None
    hpo_config: HPOConfig = None
    vllm_config: VLLMGenerationConfig = None
    shot_task_generation_config = {}
    cot_task_generation_config = {}
    model: str = "mistral-7b-inst-3"
    quantization: bool = False


    @staticmethod
    def register_cli(arg_parser):
        """
        Registers all cli parameters for RunConfig
        """
        arg_parser.add_argument("--quantization", action="store_true")
        arg_parser.add_argument("--sample", action="store_true")
        arg_parser.add_argument("-icot", "--chain_of_thoughts", action="store_true")

        arg_parser.add_argument("-int", "--in_task", type=bool, help="whether to conduct a cross task or in task experiment")
        arg_parser.add_argument("-sf", "--skill_filter", type=str, help="filter the tasks based on skill")
        arg_parser.add_argument("-d", "--debug", action="store_true", default=False, help="Should prompting be performed")
        arg_parser.add_argument("-iprpt", "--prompting", action="store_true", default=False, help="Should prompting be performed")
        arg_parser.add_argument("-ie", "--is_evaluate", action="store_true", default=False, help="Should evaluation be performed")
        arg_parser.add_argument("-ih", "--hpo", action="store_true", default=False, help="Should HPO be performed")
        arg_parser.add_argument("-s", "--seed", type=int, help="Seed to use for running experiment")
        arg_parser.add_argument("-k", "--train_subsample_rate", type=float, help="Fraction of instances to subsample from each dataset")
        arg_parser.add_argument("-tsa", "--train_subsample_amount", type=int, help="Amount of instances to subsamplea from each dataset")
        arg_parser.add_argument("-l", "--is_leave_one_out", action="store_true", help="Should leave one out training be performed")
        arg_parser.add_argument("-vsr", "--test_subsample_rate", type=float, help="Fraction of instances to subsample from each dataset for testing")
        arg_parser.add_argument("-vsa", "--test_subsample_amount", type=int, help="Amount of instances to subsamplea from each dataset for testing")
        arg_parser.add_argument("-tdm", "--test_dataset_match", type=str, help="Matching pattern for test dataset files to include")
        arg_parser.add_argument("-tdn", "--test_dataset_name", type=str, help="Name of the test dataset to use")
        arg_parser.add_argument("-rc", "--resume_checkpoint", help="Resume training from checkpoint")
        arg_parser.add_argument("-la", "--load_adapter", action="append", help="Adapter to load")
        arg_parser.add_argument("-an", "--adapter_name", action="append", help="Adapter name that is being loaded")
        arg_parser.add_argument("-co", "--config_output", type=Path, help="File to write config to")
        arg_parser.add_argument("-df", "--data_folder", type=Path, help="Data folder path")
        arg_parser.add_argument("-mf" , "--models_folder", type=Path, help="Data folder path")
        # Training arguments
        arg_parser.add_argument("-tbs", "--train_batch_size", type=int, help="Training batch size")
        arg_parser.add_argument("-te", "--train_epochs", type=int, help="Number of training epochs")
        arg_parser.add_argument("-tlr", "--train_learning_rate", type=float, help="Learning rate")
        arg_parser.add_argument("-to", "--train_optim", type=str, help="Optimizer")
        arg_parser.add_argument("-tes", "--train_evaluation_strategy", type=str, help="Evaluation strategy")
        arg_parser.add_argument("-tss", "--train_save_strategy", type=str, help="Save strategy")
        arg_parser.add_argument("-tev", "--train_eval_steps", type=int, help="Eval steps")
        arg_parser.add_argument("-tsv", "--train_save_steps", type=int, help="Save steps")
        arg_parser.add_argument("-tod", "--output_dir", type=str, help="Output directory")
        arg_parser.add_argument("-stl", "--train_save_total_limit", type=int, help="Maximum number of last checkpoint files to keep.")
        arg_parser.add_argument("-tw", "--train_warmup_steps", type=int, help="Linear warmup over warmup_steps")
        arg_parser.add_argument("-tfp", "--train_fp16", action="store_true", help="Use float16 training")
        arg_parser.add_argument("-tls", "--train_logging_steps", type=int, help="Log & save metrics to tensorboard every logging_steps steps")
        arg_parser.add_argument("-tlst", "--train_lr_scheduler_type", type=str, help="Learning rate scheduler type")
        arg_parser.add_argument("-tmfb", "--train_metric_for_best_model", type=str, help="Metric for best model selection")
        arg_parser.add_argument("-tlbme", "--train_load_best_model_at_end", action="store_true", help="Whether to load the best model at the end.")
        arg_parser.add_argument("-tgbl", "--train_group_by_length", action="store_true", help="Group sequences into batches with same length")
        arg_parser.add_argument("-tde", "--train_do_eval", action="store_true", help="Whether to run evaluation during training")
        arg_parser.add_argument("-teb", "--train_eval_batch_size", type=int, help="Batch size for eval dataset")
        arg_parser.add_argument("-tb1", "--train_adam_beta1", type=float, help="Adam optimizer beta 1 parameter")
        arg_parser.add_argument("-tb2", "--train_adam_beta2", type=float, help="Adam optimizer beta 2 parameter")
        arg_parser.add_argument("-tae", "--train_adam_epsilon", type=float, help="Adam optimizer epsilon parameter")
        arg_parser.add_argument("-tmgn", "--train_max_grad_norm", type=float, help="Gradient clipping max_grad_norm")
        arg_parser.add_argument("-ttf32", "--train_tf32", action="store_true", help="tf32 training acceleration")
        arg_parser.add_argument("-tbf16", "--train_bf16", action="store_true", help="bf16 training acceleration")
        arg_parser.add_argument("-tgc", "--train_gradient_checkpointing", action="store_true", help="Activate gradient checkpointing")
        arg_parser.add_argument("-tgas", "--train_gradient_accumulation_steps", type=int, help="Gradient accumulation steps")
        # Validation arguments
        arg_parser.add_argument("-em", "--eval_metric", type=str, help="Evaluation metric name")
        arg_parser.add_argument("-bs", "--validation_batch_size", type=int, help="Batch size for evaluation")
        arg_parser.add_argument("-fa", "--fscore_average", type=str, help="F-score average mode")
        arg_parser.add_argument("-fb", "--fscore_beta", type=float, help="Beta parameter for F-score")
        # Generation arguments
        arg_parser.add_argument("-ml", "--max_length", type=int, help="Maximum sequence length during generation")
        arg_parser.add_argument("-mn", "--max_new_tokens", type=int, help="Maximum number of new tokens to generate")
        arg_parser.add_argument("-min", "--min_length", type=int, help="Minimum sequence length during generation")
        arg_parser.add_argument("-mnn", "--min_new_tokens", type=int, help="Minimum number of new tokens to generate")
        arg_parser.add_argument("-es", "--early_stopping", help="Early stopping condition")
        arg_parser.add_argument("-mt", "--max_time", type=float, help="Maximum generation time in seconds")
        arg_parser.add_argument("-ds", "--do_sample", action="store_true", help="Use sampling method for generation")
        arg_parser.add_argument("-nb", "--num_beams", type=int, help="Number of beams for beam search")
        arg_parser.add_argument("-nbg", "--num_beam_groups", type=int, help="Number of beam groups for group beam search")
        arg_parser.add_argument("-pa", "--penalty_alpha", type=float, help="Alpha parameter for penalty function")
        arg_parser.add_argument("-uc", "--use_cache", action="store_true", help="Use cache during generation")
        arg_parser.add_argument("-temp", "--temperature", type=float, help="Temperature parameter for sampling method")
        arg_parser.add_argument("-tk", "--top_k", type=int, help="Top-k sampling parameter")
        arg_parser.add_argument("-p", "--top_p", type=float, help="Top-p (nucleus) sampling parameter")
        arg_parser.add_argument("-mp", "--min_p", type=float, help="Minimum p value for typical (Tyers) sampling")
        arg_parser.add_argument("-tp", "--typical_p", type=float, help="Typical p (Tyers) sampling parameter")
        arg_parser.add_argument("-ec", "--epsilon_cutoff", type=float,  help="Epsilon cutoff parameter")
        arg_parser.add_argument("-etac", "--eta_cutoff", type=float, help="Eta cutoff parameter")
        arg_parser.add_argument("-dp", "--diversity_penalty", type=float, help="Diversity penalty parameter")
        arg_parser.add_argument("-rp", "--repetition_penalty", type=float, help="Repetition penalty parameter")
        arg_parser.add_argument("-lp", "--length_penalty", type=float, help="Length penalty parameter")
        # Quantization config
        arg_parser.add_argument("-bm", "--base_model", type=str, help="base model")
        arg_parser.add_argument("-esp", "--experiment_splits_path", type=str)
        arg_parser.add_argument("-pp", "--prediction_path", type=str)

    @classmethod
    def from_file(cls, paths: List[Path], args=None):
        """
        Initializes RunConfig from configuration file and cli parameters

        :param paths: List of paths to RunConfig configuration files that will be compiled together
        :param args: CLI arguments to overwrite onfiguration file
        :returns: Initialized RunConfig
        """
        config = {}
        for path in paths:

            with open(path, "r") as f:
                conf_file = json.load(f)
                update_conf(config, conf_file)
        if args != None:
            if args.config:
                with open(args.config[0] , "r") as f:
                    conf_file = json.load(f)
                    update_conf(config, conf_file)
        if os.path.exists("/mnt/kisski"):
            config = rewrite_config(config, "/bigwork/nhwpajjy", "/mnt/home/yajjour")

        conf_obj =  cls(**config)
        conf_obj.shot_task_generation_config = {}

        if config.get("generation_config_path"):
            with open(config.get("generation_config_path")) as task_config_stream:
                task_specific_generation_configs = json.load(task_config_stream)
            for task in task_specific_generation_configs["shot"]:
                conf_obj.shot_task_generation_config[task] = VLLMGenerationConfig(**task_specific_generation_configs["shot"][task])
            for task in task_specific_generation_configs["chain-of-thought"]:
                conf_obj.cot_task_generation_config[task] = VLLMGenerationConfig(**task_specific_generation_configs["chain-of-thought"][task])

        if config.get("peft_configs"):
            peft_configs = []
            for conf in conf_obj.peft_configs:
                peft_conf = PeftPretrainedConfig(**conf)
                if peft_conf.adapter_type:
                    config_cls = PEFT_TYPE_TO_CONFIG_MAPPING[peft_conf.adapter_type]
                    peft_conf.config = config_cls(**peft_conf.config)
                peft_configs.append(peft_conf)

            conf_obj.peft_configs = peft_configs
        if config.get("training_args_config"):
            conf_obj.training_args_config = TrainingArgsConfig(**conf_obj.training_args_config)
        if config.get("data_collator_config"):
            conf_obj.data_collator_config = DataCollatorConfig(**conf_obj.data_collator_config)
        if config.get("generation_config"):
            conf_obj.generation_config = ModelGenerationConfig(**conf_obj.generation_config)
        if config.get("validation_config"):
            conf_obj.validation_config = ValidationConfig(**conf_obj.validation_config)
        if config.get("early_stopping_config"):
            conf_obj.early_stopping_config = EarlyStoppingConfig(**conf_obj.early_stopping_config)
        if config.get("peft_fresh_config"):
            conf_obj.peft_fresh_config = PeftAdapterConfig(**conf_obj.peft_fresh_config)
        if config.get("hpo_config"):
            #set_trace()
            conf_obj.hpo_config = HPOConfig(**conf_obj.hpo_config)
        if config.get("vllm_config"):
            conf_obj.vllm_config = VLLMGenerationConfig(**conf_obj.vllm_config)
        if config.get("model_configs"):
            model_configs = []
            for conf in conf_obj.model_configs:
                model_config = ModelConfig(**conf)
                model_configs.append(model_config)
                if model_config.label == conf_obj.base_model:
                    conf_obj.model_config = model_config
            conf_obj.model_configs = model_configs


        if not args:
            return conf_obj
        # Runner config
        if args.prediction_path:
            conf_obj.prediction_path = args.prediction_path
        if args.sample:
            conf_obj.sample = True
        if args.skill_filter:
            conf_obj.skill_filter = args.skill_filter
        if args.in_task:
            conf_obj.in_task = True

        if args.debug:
            conf_obj.debug = args.debug
        if args.seed:
            conf_obj.seed = args.seed
        if args.chain_of_thoughts:
            conf_obj.chain_of_thoughts = True
        if args.prompting:
            conf_obj.prompting = True
        if args.train_subsample_rate:
            conf_obj.train_datasets["subsample_rate"] = args.train_subsample_rate
        if args.train_subsample_amount:
            conf_obj.train_datasets["subsample_amount"] = args.train_subsample_amount
        if args.is_leave_one_out:
            conf_obj.train_datasets["leave_one_out"] = args.is_leave_one_out
        if args.test_subsample_rate:
            conf_obj.test_dataset["subsample_rate"] = args.test_subsample_rate
        if args.test_subsample_amount:
            conf_obj.test_dataset["subsample_amount"] = args.test_subsample_amount
        if args.test_dataset_name:
            conf_obj.test_dataset["name"] = args.test_dataset_name
        if args.base_model:
            conf_obj.base_model = args.base_model
        ## This should be executed after choosing the model
        if args.quantization:
            conf_obj.quantization = True

        for conf in conf_obj.model_configs:

            if conf.label == conf_obj.base_model:
                conf_obj.model_config = conf


        if args.data_folder:
            conf_obj.data_folder = args.data_folder

        if args.models_folder:
            conf_obj.models_folder = args.models_folder

        if args.load_adapter:
            peft_configs = []
            for i, adapter in enumerate(args.load_adapter):
                peft_configs.append(PeftPretrainedConfig(
                    model_id=adapter,
                    adapter_name=args.adapter_name[i],
                    is_trainable=not args.is_evaluate
                ))
            conf_obj.peft_configs = peft_configs

        best_hyper_parameters: {}
        if not conf_obj.hpo and not conf_obj.prompting and config.get("best_hyper_parameters_path"):
            with open(config.get("best_hyper_parameters_path")) as hps_stream:
                best_hyper_parameters = json.load(hps_stream)
                if conf_obj.base_model in best_hyper_parameters:
                    model_hyper_parameters = best_hyper_parameters[conf_obj.base_model]
                    test_dataset = conf_obj.test_dataset["name"]
                    if test_dataset in model_hyper_parameters:
                        task_hyper_parameters = model_hyper_parameters[test_dataset]
                        conf_obj.training_args_config.per_device_train_batch_size = task_hyper_parameters["train_batch_size"]
                        conf_obj.training_args_config.learning_rate = task_hyper_parameters["learning_rate"]
                        print(f"loading best params for  {test_dataset} and {conf_obj.base_model}")
                        print(f"setting learning rate {conf_obj.training_args_config.learning_rate} and batch size {conf_obj.training_args_config.per_device_train_batch_size}")
                    else:
                        print(f"Error no params found for {test_dataset} and {conf_obj.base_model}")
                else:
                    print(f"Error no params found for {conf_obj.base_model}")

        # Training arguments
        if args.train_batch_size:
            conf_obj.training_args_config.per_device_train_batch_size = args.train_batch_size
        if args.train_epochs:
            conf_obj.training_args_config.num_train_epochs = args.train_epochs
        if args.train_learning_rate:
            conf_obj.training_args_config.learning_rate = args.train_learning_rate
        if args.train_optim:
            conf_obj.training_args_config.optim = args.train_optim
        if args.train_evaluation_strategy:
            conf_obj.training_args_config.evaluation_strategy = args.train_evaluation_strategy
        if args.train_save_strategy:
            conf_obj.training_args_config.save_strategy = args.train_save_strategy
        if args.train_eval_steps:
            conf_obj.training_args_config.eval_steps = args.train_eval_steps
        if args.train_save_steps:
            conf_obj.training_args_config.save_steps = args.train_save_steps
        if args.output_dir:
            conf_obj.output_dir = args.output_dir
        if args.train_save_total_limit:
            conf_obj.training_args_config.save_total_limit = args.train_save_total_limit
        if args.train_warmup_steps:
            conf_obj.training_args_config.warmup_steps = args.train_warmup_steps
        if args.train_fp16:
            conf_obj.training_args_config.fp16 = args.train_fp16
        if args.train_logging_steps:
            conf_obj.training_args_config.logging_steps = args.train_logging_steps
        if args.train_lr_scheduler_type:
            conf_obj.training_args_config.lr_scheduler_type = args.train_lr_scheduler_type
        if args.train_metric_for_best_model:
            conf_obj.training_args_config.metric_for_best_model = args.train_metric_for_best_model
        if args.train_load_best_model_at_end:
            conf_obj.training_args_config.load_best_model_at_end = args.train_load_best_model_at_end
        if args.train_group_by_length:
            conf_obj.training_args_config.group_by_length = args.train_group_by_length
        if args.train_do_eval:
            conf_obj.training_args_config.do_eval = args.train_do_eval
        if args.train_eval_batch_size:
            conf_obj.training_args_config.per_device_eval_batch_size = args.train_eval_batch_size
        if args.train_adam_beta1:
            conf_obj.training_args_config.adam_beta1 = args.train_adam_beta1
        if args.train_adam_beta2:
            conf_obj.training_args_config.adam_beta2 = args.train_adam_beta2
        if args.train_adam_epsilon:
            conf_obj.training_args_config.adam_epsilon
        if args.train_max_grad_norm:
            conf_obj.training_args_config.max_grad_norm = args.train_max_grad_norm
        if args.train_tf32:
            conf_obj.training_args_config.tf32 = args.train_tf32
        if args.train_bf16:
            conf_obj.training_args_config.bf16 = args.train_bf16
        if args.train_gradient_checkpointing:
            conf_obj.training_args_config.gradient_checkpointing = args.train_gradient_checkpointing
        if args.train_gradient_accumulation_steps:
            conf_obj.training_args_config.gradient_accumulation_steps = args.train_gradient_accumulation_steps

        # Evaluation arguments
        if args.eval_metric:
            conf_obj.validation_config.eval_metric = args.eval_metric
        if args.validation_batch_size:
            conf_obj.validation_config.validation_batch_size = args.validation_batch_size
        if args.fscore_average:
            conf_obj.validation_config.fscore_average = args.fscore_average
        if args.fscore_beta:
            conf_obj.validation_config.fscore_beta = args.fscore_beta

        # Generation arguments
        if args.max_length:
            conf_obj.cutoff_len= args.max_length
        if args.max_new_tokens:
            conf_obj.generation_config.max_new_tokens = args.max_new_tokens
        if args.min_length:
            conf_obj.generation_config.min_length = args.min_length
        if args.min_new_tokens:
            conf_obj.generation_config.min_new_tokens = args.min_new_tokens
        if args.early_stopping:
            conf_obj.generation_config.early_stopping = args.early_stopping
        if args.max_time:
            conf_obj.generation_config.max_time = args.max_time
        if args.do_sample:
            conf_obj.generation_config.do_sample = args.do_sample
        if args.num_beams:
            conf_obj.generation_config.num_beams = args.num_beams
        if args.num_beam_groups:
            conf_obj.generation_config.num_beam_groups = args.num_beam_groups
        if args.penalty_alpha:
            conf_obj.generation_config.penalty_alpha = args.penalty_alpha
        if args.use_cache:
            conf_obj.generation_config.use_cache = args.use_cache
        if args.temperature:
            conf_obj.generation_config.temperature = args.temperature
        if args.top_k:
            conf_obj.generation_config.top_k = args.top_k
        if args.top_p:
            conf_obj.generation_config.top_p = args.top_p
        if args.min_p:
            conf_obj.generation_config.min_p = args.min_p
        if args.typical_p:
            conf_obj.generation_config.typical_p = args.typical_p
        if args.epsilon_cutoff:
            conf_obj.generation_config.epsilon_cutoff = args.epsilon_cutoff
        if args.eta_cutoff:
            conf_obj.generation_config.eta_cutoff = args.eta_cutoff
        if args.diversity_penalty:
            conf_obj.generation_config.diversity_penalty = args.diversity_penalty
        if args.repetition_penalty:
            conf_obj.generation_config.repetition_penalty = args.repetition_penalty
        if args.length_penalty:
            conf_obj.generation_config.length_penalty = args.length_penalty


        if args.experiment_splits_path:
            conf_obj.experiment_splits_path = args.expreiment_splits_path



        return conf_obj


    def get_model_path(self,time):

        experiment_name = self.get_experiment_name()
        tag = f"{experiment_name}-{time}"

        return os.path.join(self.models_folder, tag)

    def get_log_path(self):
        now = datetime.now()
        starting_time = now.strftime("%m-%d-%H:%M:%S")
        if os.path.exists("/bigwork/nhwpajjy/"):
            root_path = "/bigwork/nhwpajjy/"
        else:
            root_path = "/mnt/home/yajjour"

        if self.prompting:
            self.log_path = f"{root_path}/task-specific-argument-mining-and-generation-data/logs/prompting-{self.base_model}-{starting_time}.log"
        else:
            test_dataset_name = self.test_dataset["name"]
            self.log_path = f"{root_path}/task-specific-argument-mining-and-generation-data/logs/fine-tuning-{test_dataset_name}-{self.base_model}-{starting_time}.log"
        return self.log_path
    def get_output_path(self):
        now = datetime.now()
        starting_time = now.strftime("%m-%d-%H:%M:%S")
        experiment_name = self.get_experiment_name()
        return f"{self.output_dir}/{experiment_name}-{starting_time}"


    def get_pad_token_id(self):
        if self.base_model == "qwen-7b":
            return None
        else:
            return self.pad_token_id

    def get_unk_token_id(self):
        if self.base_model == "qwen-7b":
            return None
        else:
            return "<unk>"

    def get_experiment_name(self):
        model = self.base_model


        if self.in_task:
            experiment = "in-task"
        else:
            if self.skill_filter:
                experiment = f"skill-transfer-{self.skill_filter}"
            else:
                experiment = "cross-task"
        if self.hpo:
            experiment = "hpo-" + experiment

        test_dataset_name = self.test_dataset["name"]
        exp = f"{model}-{experiment}-{test_dataset_name}"
        if "subsample_rate" in self.test_dataset:
            test_subsample_rate = self.test_dataset["subsample_rate"]
            return exp + f"-rate-{test_subsample_rate}"
        elif "subsample_amount" in self.test_dataset:
            test_subsample_amount = self.test_dataset["subsample_amount"]
            return exp + f"-amount-{test_subsample_amount}"
        else:
            return exp

    def get_tensorboard_log_dir(self):
        path_tensorboard_main = self.tensorboard_logs
        now = datetime.now()
        starting_time = now.strftime("%m-%d-%H:%M:%S")
        experiment_name = self.get_experiment_name()
        return f"{path_tensorboard_main}/{experiment_name}-{starting_time}"

    def get_experiment_type(self):
        if self.prompting:
            return ExperimentType.PROMPTING
        elif self.in_task:
            return ExperimentType.IN_TASK
        else:
            if self.skill_filter:
                return ExperimentType.SKILL_TRANSFER
            else:
                return ExperimentType.LEAVE_ONE_TASK

    def get_prompting_technique(self):
        if "subsample_amount" in self.train_datasets and self.train_datasets["subsample_amount"] == 1:
            return PromptingTechnique.ONE_SHOT
        elif "subsample_amount" in self.train_datasets and self.train_datasets["subsample_amount"] == 4:
            return PromptingTechnique.FOUR_SHOT
        elif self.chain_of_thoughts:
            return PromptingTechnique.COT
        else:
            return PromptingTechnique.ZERO_SHOT

    def get_leaderboard_path(self):
        if os.path.exists("/bigwork/nhwpajjy/"):
            path = "/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/runs"
        else:
            path = "/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/runs"

        if self.prompting:
            path = f"{path}/prompting-{self.base_model}-results.csv"
        elif self.in_task:
            path = f"{path}/in-task-{self.base_model}-results.csv"
        else:
            if self.skill_filter:
                path = f"{path}/skill-transfer-{self.base_model}-results.csv"
            else:
                path = f"{path}/cross-task-{self.base_model}-results.csv"

        return path