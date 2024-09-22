
# Running model training and evaluation

Main experiment running file `run.py` can run training and evaluation experiments with provided config:

``` shell
$ python run.py -c configs/config_name.json
```

usage: run.py [-h] [-c CONFIG] [-ie] [-s SEED] [-is INCLUDE_SUBAREA] [-ig INCLUDE_GENERES]
              [-tsr TRAIN_SUBSAMPLE_RATE] [-tsa TRAIN_SUBSAMPLE_AMOUNT] [-l]
              [-vsr TEST_SUBSAMPLE_RATE] [-vsa TEST_SUBSAMPLE_AMOUNT] [-tdm TEST_DATASET_MATCH]
              [-tdn TEST_DATASET_NAME] [-rc RESUME_CHECKPOINT] [-lm LOAD_MODEL] [-la LOAD_ADAPTER]
              [-co CONFIG_OUTPUT] [-df DATA_FOLDER] [-tbs TRAIN_BATCH_SIZE] [-te TRAIN_EPOCHS]
              [-tlr TRAIN_LEARNING_RATE] [-to TRAIN_OPTIM] [-tes TRAIN_EVALUATION_STRATEGY]
              [-tss TRAIN_SAVE_STRATEGY] [-tev TRAIN_EVAL_STEPS] [-tsv TRAIN_SAVE_STEPS]
              [-tod TRAIN_OUTPUT_DIR] [-stl TRAIN_SAVE_TOTAL_LIMIT] [-tw TRAIN_WARMUP_STEPS] [-tfp]
              [-tls TRAIN_LOGGING_STEPS] [-tlst TRAIN_LR_SCHEDULER_TYPE]
              [-tmfb TRAIN_METRIC_FOR_BEST_MODEL] [-tlbme] [-tgbl] [-tde] [-em EVAL_METRIC]
              [-bs VALIDATION_BATCH_SIZE] [-fa FSCORE_AVERAGE] [-fb FSCORE_BETA] [-ml MAX_LENGTH]
              [-mn MAX_NEW_TOKENS] [-min MIN_LENGTH] [-mnn MIN_NEW_TOKENS] [-es EARLY_STOPPING]
              [-mt MAX_TIME] [-ds] [-nb NUM_BEAMS] [-nbg NUM_BEAM_GROUPS] [-pa PENALTY_ALPHA] [-uc]
              [-temp TEMPERATURE] [-k TOP_K] [-p TOP_P] [-mp MIN_P] [-tp TYPICAL_P]
              [-ec EPSILON_CUTOFF] [-etac ETA_CUTOFF] [-dp DIVERSITY_PENALTY] [-rp REPETITION_PENALTY]
              [-lp LENGTH_PENALTY] [-i8] [-i4] [-i8t LLM_INT8_THRESHOLD] [-i8s LLM_INT8_SKIP_MODULES]
              [-f32o] [-f16w] [-4qt {fp4,nf4}] [-dq]


Run peft finetuning experiment

options:
  **-h**, **--help**           show this help message and exit

  **-c CONFIG, --config CONFIG** Path to experiment config

  **-ie, --is_evaluate**    Should only evaluation be performed

  **-s SEED, --seed SEED**  Seed to use for running experiment

  **-is INCLUDE_SUBAREA, --include_subarea INCLUDE_SUBAREA** Training set subareas

  **-ig INCLUDE_GENERES, --include_generes INCLUDE_GENERES** Training set genres

  **-tsr TRAIN_SUBSAMPLE_RATE, --train_subsample_rate TRAIN_SUBSAMPLE_RATE** Fraction of instances to subsample from each dataset

  **-tsa TRAIN_SUBSAMPLE_AMOUNT, --train_subsample_amount TRAIN_SUBSAMPLE_AMOUNT** Amount of instances to subsamplea from each dataset

  **-l, --is_leave_one_out** Should leave one out training be performed

  **-vsr TEST_SUBSAMPLE_RATE, --test_subsample_rate TEST_SUBSAMPLE_RATE** Fraction of instances to subsample from each dataset for testing

  **-vsa TEST_SUBSAMPLE_AMOUNT, --test_subsample_amount TEST_SUBSAMPLE_AMOUNT** Amount of instances to subsamplea from each dataset for testing

  **-tdm TEST_DATASET_MATCH, --test_dataset_match TEST_DATASET_MATCH** Matching pattern for test dataset files to include

  **-tdn TEST_DATASET_NAME, --test_dataset_name TEST_DATASET_NAME** Name of the test dataset to use

  **-rc RESUME_CHECKPOINT, --resume_checkpoint RESUME_CHECKPOINT** Resume training from checkpoint

  **-lm LOAD_MODEL, --load_model LOAD_MODEL** Model to load

  **-la LOAD_ADAPTER, --load_adapter LOAD_ADAPTER** Adapter to load

  **-co CONFIG_OUTPUT, --config_output CONFIG_OUTPUT** File to write config to

  **-df DATA_FOLDER, --data_folder DATA_FOLDER** Data folder path

  **-tbs TRAIN_BATCH_SIZE, --train_batch_size TRAIN_BATCH_SIZE** Training batch size

  **-te TRAIN_EPOCHS, --train_epochs TRAIN_EPOCHS** Number of training epochs

  **-tlr TRAIN_LEARNING_RATE, --train_learning_rate TRAIN_LEARNING_RATE** Learning rate

  **-to TRAIN_OPTIM, --train_optim TRAIN_OPTIM** Optimizer

  **-tes TRAIN_EVALUATION_STRATEGY, --train_evaluation_strategy TRAIN_EVALUATION_STRATEGY** Evaluation strategy

  **-tss TRAIN_SAVE_STRATEGY, --train_save_strategy TRAIN_SAVE_STRATEGY** Save strategy

  **-tev TRAIN_EVAL_STEPS, --train_eval_steps TRAIN_EVAL_STEPS** Eval steps

  **-tsv TRAIN_SAVE_STEPS, --train_save_steps TRAIN_SAVE_STEPS** Save steps

  **-tod TRAIN_OUTPUT_DIR, --train_output_dir TRAIN_OUTPUT_DIR** Output directory

  **-stl TRAIN_SAVE_TOTAL_LIMIT, --train_save_total_limit TRAIN_SAVE_TOTAL_LIMIT** Maximum number of last checkpoint files to keep.

  **-tw TRAIN_WARMUP_STEPS, --train_warmup_steps TRAIN_WARMUP_STEPS** Linear warmup over warmup_steps

  **-tfp, --train_fp16**    Use float16 training

  **-tls TRAIN_LOGGING_STEPS, --train_logging_steps TRAIN_LOGGING_STEPS** Log & save metrics to tensorboard every logging_steps steps

  **-tlst TRAIN_LR_SCHEDULER_TYPE, --train_lr_scheduler_type TRAIN_LR_SCHEDULER_TYPE** Learning rate scheduler type

  **-tmfb TRAIN_METRIC_FOR_BEST_MODEL, --train_metric_for_best_model TRAIN_METRIC_FOR_BEST_MODEL** Metric for best model selection

  **-tlbme, --train_load_best_model_at_end** Whether to load the best model at the end.

  **-tgbl, --train_group_by_length** Group sequences into batches with same length

  **-tde, --train_do_eval** Whether to run evaluation during training

  **-em EVAL_METRIC, --eval_metric EVAL_METRIC** Evaluation metric name

  **-bs VALIDATION_BATCH_SIZE, --validation_batch_size VALIDATION_BATCH_SIZE** Batch size for evaluation

  **-fa FSCORE_AVERAGE, --fscore_average FSCORE_AVERAGE** F-score average mode

  **-fb FSCORE_BETA, --fscore_beta FSCORE_BETA** Beta parameter for F-score

  **-ml MAX_LENGTH, --max_length MAX_LENGTH** Maximum sequence length during generation

  **-mn MAX_NEW_TOKENS, --max_new_tokens MAX_NEW_TOKENS** Maximum number of new tokens to generate

  **-min MIN_LENGTH, --min_length MIN_LENGTH** Minimum sequence length during generation

  **-mnn MIN_NEW_TOKENS, --min_new_tokens MIN_NEW_TOKENS** Minimum number of new tokens to generate

  **-es EARLY_STOPPING, --early_stopping EARLY_STOPPING** Early stopping condition

  **-mt MAX_TIME, --max_time MAX_TIME** Maximum generation time in seconds

  **-ds, --do_sample**      Use sampling method for generation

  **-nb NUM_BEAMS, --num_beams NUM_BEAMS** Number of beams for beam search

  **-nbg NUM_BEAM_GROUPS, --num_beam_groups NUM_BEAM_GROUPS** Number of beam groups for group beam search

  **-pa PENALTY_ALPHA, --penalty_alpha PENALTY_ALPHA** Alpha parameter for penalty function

  **-uc, --use_cache**      Use cache during generation

  **-temp TEMPERATURE, --temperature TEMPERATURE** Temperature parameter for sampling method

  **-k TOP_K, --top_k TOP_K** Top-k sampling parameter

  **-p TOP_P, --top_p TOP_P** Top-p (nucleus) sampling parameter

  **-mp MIN_P, --min_p MIN_P** Minimum p value for typical (Tyers) sampling

  **-tp TYPICAL_P, --typical_p TYPICAL_P** Typical p (Tyers) sampling parameter

  **-ec EPSILON_CUTOFF, --epsilon_cutoff EPSILON_CUTOFF** Epsilon cutoff parameter

  **-etac ETA_CUTOFF, --eta_cutoff ETA_CUTOFF** Eta cutoff parameter

  **-dp DIVERSITY_PENALTY, --diversity_penalty DIVERSITY_PENALTY** Diversity penalty parameter

  **-rp REPETITION_PENALTY, --repetition_penalty REPETITION_PENALTY** Repetition penalty parameter

  **-lp LENGTH_PENALTY, --length_penalty LENGTH_PENALTY** Length penalty parameter

  **-i8, --load_in_8bit**   Load model in 8-bit precision

  **-i4, --load_in_4bit**   Load model in 4-bit precision

  **-i8t LLM_INT8_THRESHOLD, --llm_int8_threshold LLM_INT8_THRESHOLD** LLM Int8 threshold for quantization

  **-i8s LLM_INT8_SKIP_MODULES, --llm_int8_skip_modules LLM_INT8_SKIP_MODULES** List of modules to skip when using LLM Int8 quantization

  **-f32o, --enable_fp32_cpu_offload** Enable FP32 CPU offloading for LLM Int8

  **-f16w, --has_fp16_weight** Set if the model has FP16 weights for LLM Int8

  **-4qt {fp4,nf4}, --bnb_4bit_quant_type {fp4,nf4}** BitsAndBytes 4-bit quantization type

  **-dq, --double_quant**   Enable double quantization for BitsAndBytes 4-bit

## Running config

Running configuration should follow `RunConfig` data class in [config.py](config.py) in form of json. One json config can be shared between many runs specified by `.json` files.


## Changing Training data

### Train set

In order to include datasets into training set from genres `train_datasets.include_genres` parameter need to be set.

Multiple configuration files can be combined together

``` shell
$ python run.py -c configs/config_name.json -c configs/custom_config.json
```


``` json
{
    "train_datasets": {
        "include_genres": ["genre"]
    }
}

```


In order to include datasets into training set from subareas `train_datasets.include_subareas` parameter need to be set.

``` json
{
    "train_datasets": {
        "include_subareas": ["subarea"]
    }
}

```

### Test set

Test dataset must be specified separatedly in `test_datasets` field.

``` json
"dataset_name": {
    "name": "dataset_folder_name",
    "match": "dataset_file_matrch",
    "prompt_template": "### Instruction:\n{definition}\n### Input: {instance_input}\n### Response:",
    "subsample_amount": 50
}
```
