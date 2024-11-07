from dataclasses import dataclass, field
from typing import List
from pathlib import Path
import json

from optuna import Trial
from peft.config import PeftConfig
from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING

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

    evaluation_strategy: str

    save_strategy: str

    eval_steps: int

    save_steps: int

    output_dir: str

    save_total_limit: int = 3

    weight_decay: float = 0.0

    per_device_eval_batch_size: int = 8

    adam_beta1: float = 0.9

    adam_beta2: float = 0.999

    adam_epsilon: float = 1e-8

    max_grad_norm: float = 1.0

    warmup_steps: int = None

    fp16: bool = True

    logging_steps: int = 10

    lr_scheduler_type: str = "constant_with_warmup"

    metric_for_best_model: str = None

    tf32: bool = False

    bf16: bool = False

    gradient_checkpointing: bool = False

    gradient_accumulation_steps: int = 1

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
class QuantConfig(CommonConfig):
    """Configuration object for BitsAndBytesConfig"""

    load_in_8bit: bool = False

    load_in_4bit: bool = False

    llm_int8_threshold: float = 6.0

    llm_int8_skip_modules: List[str] = None

    llm_int8_enable_fp32_cpu_offload: bool = False

    llm_int8_has_fp16_weight: bool = False

    bnb_4bit_quant_type: str = "fp4"

    bnb_4bit_use_double_quant: bool = False

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

    val_metric: str = None

    llama_causal_config: dict = field(default_factory=dict)
    quant_config: dict = field(default_factory=dict)
    # Training configs
    training_args_config: dict = field(default_factory=dict)
    early_stopping_config: dict = field(default_factory=dict)
    data_collator_config: dict = field(default_factory=dict)

@dataclass
class RunConfig:
    """Config for instruction finetuning run"""

    # Seed to use
    seed: int
    # Prompt cutoff length
    cutoff_len: int
    # Training datasets
    train_datasets: dict
    # Test datasets
    test_datasets: dict
    # Data folder
    data_folder: str
    # Base model path
    base_model: str
    # Should only evaluation be performed
    is_eval: bool
    # Should HPO be performed
    is_hpo: bool
    # Run config path
    run_output_path: str
    # Padding token id
    pad_token_id: int = 0
    # Peft combination type
    combination_type: str = None
    # Data type
    data_type: str = "ndjson"

    llama_causal_config: LLamaCausalConfig = None
    quant_config: QuantConfig = None
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

    @staticmethod
    def register_cli(arg_parser):
        """
        Registers all cli parameters for RunConfig
        """
        arg_parser.add_argument("-ie", "--is_evaluate", action="store_true", default=False, help="Should evaluation be performed")
        arg_parser.add_argument("-ih", "--is_hpo", action="store_true", default=False, help="Should HPO be performed")
        arg_parser.add_argument("-s", "--seed", type=int, help="Seed to use for running experiment")
        arg_parser.add_argument("-is", "--include_subarea", action="append", help="Training set subareas")
        arg_parser.add_argument("-ig", "--include_generes", action="append", help="Training set genres")
        arg_parser.add_argument("-tsr", "--train_subsample_rate", type=float, help="Fraction of instances to subsample from each dataset")
        arg_parser.add_argument("-tsa", "--train_subsample_amount", type=int, help="Amount of instances to subsamplea from each dataset")
        arg_parser.add_argument("-l", "--is_leave_one_out", action="store_true", help="Should leave one out training be performed")
        arg_parser.add_argument("-vsr", "--test_subsample_rate", type=float, help="Fraction of instances to subsample from each dataset for testing")
        arg_parser.add_argument("-vsa", "--test_subsample_amount", type=int, help="Amount of instances to subsamplea from each dataset for testing")
        arg_parser.add_argument("-tdm", "--test_dataset_match", type=str, help="Matching pattern for test dataset files to include")
        arg_parser.add_argument("-tdn", "--test_dataset_name", type=str, help="Name of the test dataset to use")
        arg_parser.add_argument("-rc", "--resume_checkpoint", help="Resume training from checkpoint")
        arg_parser.add_argument("-lm", "--load_model", type=str, help="Model to load")
        arg_parser.add_argument("-la", "--load_adapter", action="append", help="Adapter to load")
        arg_parser.add_argument("-an", "--adapter_name", action="append", help="Adapter name that is being loaded")
        arg_parser.add_argument("-co", "--config_output", type=Path, help="File to write config to")
        arg_parser.add_argument("-df", "--data_folder", type=Path, help="Data folder path")
        # Training arguments
        arg_parser.add_argument("-tbs", "--train_batch_size", type=int, help="Training batch size")
        arg_parser.add_argument("-te", "--train_epochs", type=int, help="Number of training epochs")
        arg_parser.add_argument("-tlr", "--train_learning_rate", type=float, help="Learning rate")
        arg_parser.add_argument("-to", "--train_optim", type=str, help="Optimizer")
        arg_parser.add_argument("-tes", "--train_evaluation_strategy", type=str, help="Evaluation strategy")
        arg_parser.add_argument("-tss", "--train_save_strategy", type=str, help="Save strategy")
        arg_parser.add_argument("-tev", "--train_eval_steps", type=int, help="Eval steps")
        arg_parser.add_argument("-tsv", "--train_save_steps", type=int, help="Save steps")
        arg_parser.add_argument("-tod", "--train_output_dir", type=str, help="Output directory")
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
        arg_parser.add_argument("-k", "--top_k", type=int, help="Top-k sampling parameter")
        arg_parser.add_argument("-p", "--top_p", type=float, help="Top-p (nucleus) sampling parameter")
        arg_parser.add_argument("-mp", "--min_p", type=float, help="Minimum p value for typical (Tyers) sampling")
        arg_parser.add_argument("-tp", "--typical_p", type=float, help="Typical p (Tyers) sampling parameter")
        arg_parser.add_argument("-ec", "--epsilon_cutoff", type=float,  help="Epsilon cutoff parameter")
        arg_parser.add_argument("-etac", "--eta_cutoff", type=float, help="Eta cutoff parameter")
        arg_parser.add_argument("-dp", "--diversity_penalty", type=float, help="Diversity penalty parameter")
        arg_parser.add_argument("-rp", "--repetition_penalty", type=float, help="Repetition penalty parameter")
        arg_parser.add_argument("-lp", "--length_penalty", type=float, help="Length penalty parameter")
        # Quantization config
        arg_parser.add_argument("-i8", "--load_in_8bit", action="store_true", help="Load model in 8-bit precision")
        arg_parser.add_argument("-i4", "--load_in_4bit", action="store_true", help="Load model in 4-bit precision")
        arg_parser.add_argument("-i8t", "--llm_int8_threshold", type=float, help="LLM Int8 threshold for quantization")
        arg_parser.add_argument("-i8s", "--llm_int8_skip_modules", action="append", help="List of modules to skip when using LLM Int8 quantization")
        arg_parser.add_argument("-f32o", "--enable_fp32_cpu_offload", action="store_true", help="Enable FP32 CPU offloading for LLM Int8")
        arg_parser.add_argument("-f16w", "--has_fp16_weight", action="store_true", help="Set if the model has FP16 weights for LLM Int8")
        arg_parser.add_argument("-4qt", "--bnb_4bit_quant_type", choices=["fp4", "nf4"], help="BitsAndBytes 4-bit quantization type")
        arg_parser.add_argument("-dq", "--double_quant", action="store_true", help="Enable double quantization for BitsAndBytes 4-bit")



    @classmethod
    def from_file(cls, paths: List[Path], args):
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

        conf_obj = cls(is_eval=args.is_evaluate, is_hpo=args.is_hpo, **config)

        if config.get("llama_causal_config"):
            conf_obj.llama_causal_config = LLamaCausalConfig(**conf_obj.llama_causal_config)
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
        if config.get("quant_config"):
            conf_obj.quant_config = QuantConfig(**conf_obj.quant_config)
        if config.get("hpo_config"):
            conf_obj.hpo_config = HPOConfig(**conf_obj.hpo_config)


        # Runner config
        if args.seed:
            conf_obj.seed = args.seed
        if args.include_subarea:
            conf_obj.train_datasets["include_subarea"] = args.include_subarea
        if args.include_generes:
            conf_obj.train_datasets["include_genres"] = args.include_generes
        if args.train_subsample_rate:
            conf_obj.train_datasets["subsample_rate"] = args.train_subsample_rate
        if args.train_subsample_amount:
            conf_obj.train_datasets["subsample_amount"] = args.train_subsample_amount
        if args.is_leave_one_out:
            conf_obj.train_datasets["leave_one_out"] = args.is_leave_one_out
        if args.test_subsample_rate:
            conf_obj.test_datasets["subsample_rate"] = args.test_subsample_rate
        if args.test_subsample_amount:
            conf_obj.test_datasets["subsample_amount"] = args.test_subsample_amount
        if args.test_dataset_match:
            conf_obj.test_datasets["match"] = args.test_dataset_match
        if args.test_dataset_name:
            conf_obj.test_datasets["name"] = args.test_dataset_name
        if args.load_model:
            conf_obj.base_model = args.load_model
        if args.config_output:
            conf_obj.run_output_path = args.config_output
        if args.data_folder:
            conf_obj.data_folder = args.data_folder
        if args.load_adapter:
            peft_configs = []
            for i, adapter in enumerate(args.load_adapter):
                peft_configs.append(PeftPretrainedConfig(
                    model_id=adapter,
                    adapter_name=args.adapter_name[i],
                    is_trainable=not args.is_evaluate
                ))
            conf_obj.peft_configs = peft_configs

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
        if args.train_output_dir:
            conf_obj.training_args_config.output_dir = args.train_output_dir
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
            conf_obj.generation_config.max_length = args.max_length
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

        # Quantization config
        if args.load_in_8bit:
            conf_obj.quant_config.load_in_8bit = args.load_in_8bit
        if args.load_in_4bit:
            conf_obj.quant_config.load_in_4bit = args.load_in_4bit
        if args.llm_int8_threshold:
            conf_obj.quant_config.llm_int8_threshold = args.llm_int8_threshold
        if args.llm_int8_skip_modules:
            conf_obj.quant_config.llm_int8_skip_modules = args.llm_int8_skip_modules
        if args.enable_fp32_cpu_offload:
            conf_obj.quant_config.enable_fp32_cpu_offload = args.enable_fp32_cpu_offload
        if args.has_fp16_weight:
            conf_obj.quant_config.has_fp16_weight = args.has_fp16_weight
        if args.bnb_4bit_quant_type:
            conf_obj.quant_config.bnb_4bit_quant_type = args.bnb_4bit_quant_type
        if args.double_quant:
            conf_obj.quant_config.double_quant = args.double_quant

        return conf_obj
