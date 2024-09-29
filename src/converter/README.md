# Converter scripts

This folder contains a scirpt to process each dataset.

Each dataset has to conform to a common naming convention: `convert_[task]_[dataset]_[modifier]_[paper_author][year].py`. Also preprocessing scripts should use common API from [common.py](./common.py) file. Example usage of how to structure a dataset preprocessing file can be found [in a template file](./convert_template.py).

In order to make changing of output datat folder easier, it is recommended to use `Output` class API to write output data and files onto disk. Output repository path can be set using [config.yaml](../../config.yaml) `data_repo` parameter.

Every dataset should have a record in a `metadata.json` file to provide easy access to contents and additional information of each file. Metadata can be added to a file using `Metadata` class in [common.py](./common.py).

## Checking prompt size

Output datasets prompt size can be checked using [prompt_size_counter.py](./prompt_size_counter.py) script. 

Check character size:

``` shell
$ python prompt_size_counter.py
```

Check token size with `LlamaTokenizer`:

``` shell
$ python prompt_size_counter.py --tokenizer_name baffo32/decapoda-research-llama-7B-hf --tokenizer_type llama_tokenizer
```

## Testing instance amount

To test if instance amount of output file is the same as what was reported in the original paper, a test file is used:

``` shell
$ pytest test_dataset_canonical_size.py
```

Canonical file sizes can be set in [dataset_canonical_size.json](./dataset_canonical_size.json) config file.

``` json-with-comments
{
    {
        "name": "argument_unit_segmentation_ajjour17", // Dataset name
        "file_list": [ // Files to include
            "argument_unit_segmentation_web_discourse_test_ajjour17.json",
            "argument_unit_segmentation_web_discourse_train_ajjour17.json"
        ],
        "canon_size": 340 // Amount of instances in paper
    }
}
```

