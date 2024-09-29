# Preprocessing data

In order to run experiments, data must be preprocessed first into `ndjson` files.

``` shell
$ python process_jsonl.py -o /output/folder/
```

# Running experiment

## Running with configuration file

Main experiment running file `run.py` can run training and evaluation experiments with provided config file:

``` shell
$ python run.py -c configs/config_name.json
```

Multiple configuration files can be composed together. Values from later configuration files will overwrite the previous configuration files

``` shell
$ python run.py -c configs/general_config.json -c configs/specific_config.json
```

#### Examples

Running a single configuration:

``` shell
$ python run.py -c configs/config_debates_key_point_barhaim21.json
```

Running from composed config:

``` shell
python run.py -c configs/config_leave_one_out.json -c configs/config_leave_one_out_barhaim17.json
```

## Setting configuration values in command line

Many experiment configuration values can be set using command line flags.

### Evaluate model without training

Skips training the model and evaluates it.

``` shell
$ python run.py -c configs/config_file.json --is_evaluate
```

#### Example

Skip training and evaluate config:

``` shell
$ python run.py -c configs/config_debates_key_point_barhaim21.json --is_evaluate
```

### Set input/output folders

Path to the output file that contains experiment evaluation results

``` shell
$ pyhton run.py -c configs/config_file.json --config_output /path/to/output.json
```

Folder that contains preprocessed data to be used in experiment.

``` shell
$ pyhton run.py -c configs/config_file.json --data_folder /path/to/data/folder/
```

#### Example

``` shell
$ python run.py -c configs/config_debates_key_point_barhaim21.json --data_folder /home/dima/Projects/data/ --config_output ./output.json
```

### Load model checkpoints

Adapter can be loaded using `--load_adapter` flag with path to adapter folder or name.

``` shell
$ pyhton run.py -c configs/config_file.json --load_adapter /path/to/adapter/folder/

```

Model can be loaded using `--load_model` flag with flag to model folder or name.

``` shell
$ pyhton run.py -c configs/config_file.json --load_model /path/to/adapter/folder/
```

#### Example

Load standard llama model for finetuning

``` shell
$ python run.py -c configs/config_debates_key_point_barhaim21.json --load_model baffo32/decapoda-research-llama-7B-hf --load_adapter tloen/alpaca-lora-7b
```

### Test set settings

Test dataset name can be specified with `--test_dataset_name` flag while dataset files to match inside dataset folder can be set using `--test_dataset_match` flag. If full evaluation of test datasets is not needed, it can be subsampled using `--test_subsample_amount` to subsample only `amount_sample` samples, or `--test_subsample_rate` to subsample a fraction `ratio_subsample` of datapoints.

``` shell
$ python run.py -c configs/config_file.json --test_dataset_name dataset_name --test_dataset_match match_file_string --test_subsample_amount amount_subsample --test_subsample_rate ratio_subsample
```

#### Example

Use same config but with different test set with 50 datapoints:

``` shell
$ python run.py -c configs/config_debates_key_point_barhaim21.json --test_dataset_name argument_unit_classificaiton_wikipedia_articles_lexisnexis_eindor20 --test_dataset_match argument_unit_classificaiton_wikipedia_articles_lexisnexis_eindor20 --test_subsample_amount 50
```

### Train set settings

With setting `--is_leave_one_out` all datasets except test set are taken into training set. In order to subsample each dataset `--train_subsample_amount` and `--train_subsample_rate` is used that function the same as with test set.

``` shell
$ python run.py -c configs/config_file.json --is_leave_one_out --train_subsample_rate subsample_rate --train_subsample_amount subsample_amount
```

If you only want to include certain subareas of datasets into training set `--include_subarea` can be used. 

``` shell
$ python run.py -c configs/config_file.json --include_subarea subarea
```

If you want to include certain genres of datasets into training set `--include_generes` can be used.

``` shell
$ python run.py -c configs/config_file.json --include_genres genres
```

### Frequently used hyperparameters

Most hyperparameters can be also changed using command line arguments. The naming convention of them follows the parameter names of huggingface transformers API.

``` shell
$ python run.py -c configs/config_file.json --train_batch_size batch_size --train_epochs train_epochs --train_learning_rate learning_rate
```

#### Other hyperparameters

You can find out about other parameters and hyperparameters using `--help` flag.

``` shell
$ pyhton run.py --help
```

### Examples

## Examples of running experiments

Running [leave one out](./evaluation.md) experiment evaluation:

``` shell
$ python run.py -c configs/config_leave_one_out.json -c configs/config_leave_one_out_barhaim17.json -co out_barhaim17.json -ie -la /home/dima/Projects/evaluations/leave-one-out-stance_classification_ibmsc_barhaim17/checkpoint-10/adapter/ -vsa 50
```

Running [leave one out](./evaluation.md) experiment training:

``` shell
$ python run.py -c configs/config_leave_one_out.json -c configs/config_leave_one_out_barhaim17.json -co out_barhaim17.json
```

## Running baselines

Baseline files can be configured the same as `run.py` experiment script.

### Majority Class

``` shell
$ python majority_class.py -c configs/config_debates_key_point_barhaim21.json 
```

### Random Label

``` shell
python random_labels.py -c configs/config_debates_key_point_barhaim21.json
```

# Configuration file

Running configuration should follow `RunConfig` data class in [config.py](config.py) in form of json. One json config can be shared between many runs specified by `.json` files.

## Train dataset config

Train dataset config structure:

``` json-with-comments
{
    "leave_one_out": true, // If leave_one_out is present and true, then all datasets except test one will be used in training set
    "include_genres": ["debates"], // Genres to take in training set
    "include_subarea": ["mining"], // Subareas to take in training set
    "include_task": ["argument_mining"] // Dataset with task to take
    "subsample_rate": 0.3, // % of datapoints to take
    "subsample_amount": 50, // Amount of training samples to take for each dataset file
    "prompt_template": "### Instruction:\n{definition}\n### Input: {instance_input}\n### Response:", // Template for compiling prompt
    "train_datasets": {
        "exclude_datasets": [ // List of datasets to exclude from training set
            "counter_argument_generation_candela_hua19",
            "counter_argument_generation_cmv_hua18"
        ]
    }
}
```

Test dataset config structure:

``` json-with-comments
{
    "test_datasets": {
        "name": "key_point_matching_argkp_2021_barhaim21", // Dataset name to use for testing
        "match": "key_point_matching_argkp_2021_barhaim21", // Match for files of dataset files
        "prompt_template": "### Instruction:\n{definition}\n### Input: {instance_input}\n### Response:", // Template to compile dataset prompt
        "subsample_amount": 50 // Amount of samples to take from test dataset
        "subsample_rate": 0.3 // % of datapoints to take from test dataset
    },
}
```

