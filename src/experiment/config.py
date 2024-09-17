from dataclasses import dataclass
from typing import List
from pathlib import Path
import json

class CommonConfig:
    """Methods for all config classes"""

    def to_conf(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}

@dataclass
class DataCollatorConfig(CommonConfig):
    """COnfiguration args for DataCollatorForSeq2Seq"""

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

    warmup_steps: int = None

    fp16: bool = True

    logging_steps: int = 10

    lr_scheduler_type: str = None

    metric_for_best_model: str = None

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

    is_trainable: bool = True

    adapter_weight: float = None

    def to_conf(self):
        config = super().to_conf()
        if config.get("adapter_weight"):
            del config["adapter_weight"]
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

    def to_conf(self):
        config = super().to_conf()
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
    # Base model path
    base_model: str
    # Padding token id
    pad_token_id: int = 0
    # Peft combination type
    combination_type: str = None

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

    @staticmethod
    def register_cli(arg_parser):
        arg_parser.add_argument("-it", "--is_train", action="store_true", default=True, help="Should training be performed")
        arg_parser.add_argument("-ie", "--is_evaluate", action="store_true", default=True, help="Should evaluation be performed")
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


    @classmethod
    def from_file(cls, path: Path, args):
        """Read config from file"""
        with open(path, "r") as f:
            config = json.load(f)

        conf_obj = cls(**config)

        if config.get("llama_causal_config"):
            conf_obj.llama_causal_config = LLamaCausalConfig(**conf_obj.llama_causal_config)
        if config.get("peft_configs"):
            peft_configs = []
            for conf in conf_obj.peft_configs:
                peft_configs.append(PeftPretrainedConfig(**conf))
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

        return conf_obj
