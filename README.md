# ArgBench

This repo comprises 33 datasets and 46 tasks covering 5 skills to evaluate how good are LLMs at computational argumentation tasks.
The skills are *argument mining*, *argument perspective assessment*,  *argument quality assessment*, *argument reasoning*, and *argument generation*.

## Experiments

You can evaluate your language model in Prompting, i.e. how good is a model at computational argumentation tasks. Or
how good is an LLM model at generalizing to unseen computational tasks (Leave-one-out), by evaluating it on five hold-out tasks
after training it on the remaining tasks as specific in this [split](argbench/experiment/configs/experiment_splits.json).
You can also evaluate a model's ability to transfer one skill (e.g., Quality) to another (e.g., Generation). 
The generation parameters for a model are stored in  [config_mistral_vllm_generation.json](argbench/experiment/configs/config_mistral_vllm_generation.json). 


### Add a model 

To evaluate your models, you have to add first it to [argbench/experiment/configs/model_configs.json](argbench/experiment/configs/model_configs.json)


### Prompting


```
./argbench/jobs/prompting/prompting.sh {gpu-count} {gpu-type} prompting.json {job-name} --model {model} --dataset {task}
```


A prompting experiment can be run with zero-shot, few-shot, or with chain-of-thoughts.

*Example for zero-shot:*
```
./argbench/jobs/prompting/prompting.sh 1 a100 prompting.json prompting-qwen --model qwen3-4b --dataset argument_relation_detection_echr_poudyal20 
```

*Example for 4-shots:*
```
./argbench/jobs/prompting/prompting.sh 1 a100 prompting.json prompting-qwen --model qwen3-4b --dataset argument_relation_detection_echr_poudyal20 -k 4 
```

*Example for chain-of-thought*:
```
./argbench/jobs/prompting/prompting.sh 1 a100 prompting.json prompting-qwen --model qwen3-4b --dataset argument_relation_detection_echr_poudyal20 --cot 
```

**Sample Prompting Experiments**

Experiment for a set of [target tasks](argbench/data/target_tasks.txt) can be run as follows

```
./argbench/jobs/prompting/prompting_target_tasks.sh -m {model} -t {task}
```

**All models and all prompting techniques**

run all models in the experiment in few-shot, chain-of-though, and zero-shot 

```
./argbench/jobs/prompting/prompting_all_tasks.sh -m qwen3-1.7b
```

### Fine tuning

**In-task**

To fine-tune an adapter for an LLM

1. Run hyper-parameter optimization on the validation set to get the best hyper-parameter (lr and batch size)

``` 
./argbench/jobs/in-tasks/run_in_tasks_experiment.sh 1 a100 in_task/in_task_{skill}_hpo.json --model {model} 
```

The best hyper-parameters and best performance can be then located in the configuration `in_task_{skill}_hpo.json` in `hpo_config` > `hpo_coarse_output`

The search process can be then accessed via optuna and is stored under `hpo_config` > `storage`

2. Add the model and hyper-parameters `hyper-parameters-in-task.json`
3. Run the test experiment. The jobs create results for the model will be appended to the leader board located in the leaderboard path
   which can be configured in the configuration file. 

``` 
bash argbench/jobs/in-tasks/run_in_tasks_experiment.sh 1 a100 in_task/in_task_{skill}.json --model {model} 
```


**Leave-one-out**

1. Run hyper-parameter optimization on the validation task to get the best hyper-parameter (lr and batch size)

``` 
bash argbench/jobs/cross-tasks/run_cross_tasks_experiment.sh 1 a100 cross_task/cross_task_val_hpo.json --model {model} 
```
2. Add the model and hyper-parameters `hyper-parameters-cross-task.json`
3. Run the test experiment. The jobs create results for the model will be appended to the leader board located in the leaderboard path
   which can be configured in the configuration file `cross_task_{skill}.json`.

``` 
bash argbench/jobs/cross-tasks/run_cross_tasks_experiment.sh 1 a100 cross_task/cross_task_{skill}.json --model {model} 
```

**Skill-transfer**

1. Run hyper-parameter optimization on the validation task to get the best hyper-parameter (lr and batch size)

``` 
To BE DONE
```
2. Add the model and hyper-parameters `To BE DONE`
3. Run the test experiment. The jobs create results for the model will be appended to the leader board located in the leaderboard path
   which can be configured in the configuration file `To BE DONE`.

``` 
To BE DONE 
```

The jobs create results for the model will be appended to the leader board located in the leaderboard path 
which can be configured in the configuration file

```
/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/runs
```

and the time for the job can be tracked in 
```
/bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/jobs
```

### Get aggregated results over skills

To get aggregated results of a specific model

```
python argbench/analysis/aggregate_results.py --file /mnt/home/yajjour/task-specific-argument-mining-and-generation-data/runs/prompting-gemma3-4b-results.csv  --metadata /mnt/home/yajjour/computational-argumentation-tasks-instructions/tasks/metadata.json --seed 1517  --skill-output-path . --k 4 --cot

```
# Benchmark 

A task in ArgBench is named with the following format *{task}*_*{dataset}*_*{authors}*. For example `warrant_generation_arc_chakarbarty21`.
ArgBench is preprocessed in three steps: Original Datasets -> Json > Ijson. To preprocess a single tasks or all tasks run the following:  


### Add a dataset



1. To add your dataset to the benchmark, you should create a converter script in [argbench/experiment/converter](argbench/experiment/converter)
The script should include a task definition and how each instance should be converted to completion tasks. For each dataset,
you should add a training, test, and validation set. 

2. You have also to add the following meta data in the code. 


*Metadata*: 

- Skill (whether it is *argument mining*, *argument quality assessment*, *argument perspective assessment*, *argument reasoning*, or *argument generation*)
- The genre (e.g., Scoial Media)
- Evaluation metric (e.g., F1-score or BertScore)
- The file and their corresponding split, i.e., which part of the dataset is a training or test set.
3. to add the dataset to the benchmark you have to run the following


```
sbatch argbench/jobs/generate_dataset.sh {task}
```

### Preprocess all datasets

To generate or prepare the benchmark you have to run
```
sbatch argbench/jobs/preprocess_all_datasets.sh
```

### Statistics for the datasets
To get statistics of the benchmark in terms of size, length of input and output.
```
python -m  argbench.analysis.dataset_size --tasks-to-shorten --output /bigwork/nhwpajjy/dataset-size.csv
```

### Run Configuration

The code repository is centered around one run configuration file, which specifies the model, the experiment, hyper-parameter configurations, and where the output of the run will be stored.
Here we list what each parameter in the configuration implies:


```
"sample": true,  
```
Whether to run on the sampled benchmark as specified in the paper or the all available data.
For prompting experiment sampling implies taking 1000 instances per task at maximum. For the fine-tuning experiment, sampling implies taking 10 % of the validation set, test set, and trainig and can be used for development.  
```
"prompting": false
```

Whether or run to fine-tune a LoRa adapter 
```
"hpo": true
```
Whether or run to fine-tune a hyper-parameter search using Optuna. 
```
"in_task": false
```
Whether or not to run an in-task or cross-experiment.
```
"chain_of_thoughts": false
```
whether or not to run an chain-of-thought prompting technique 
```
"experiment_splits_path": "/bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/experiment/configs/experiment_splits.json",
```
The specification of the split for the leave-one-out and in-task experiments
```
"debug": false
```
Whether or not to log key intermediate output 
```
"model": "mistral-7b-inst-3"
```
The model to run the experiment 
```
"cutoff_len": 300
```
The maximum length of the input prompt, including task definition and few shot examples. 
```
"data_folder": "/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/tasks",
```
The location of the output of the benchmark preprocessing 
```
"models_folder": "/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/fine-tuned-models",
```
The location of the offline downloaded hugging face models 
```
"data_type": "ndjson",
```
The output format of the preprocessed benchmark
```
"generation_config_path": "/bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/experiment/configs/config_mistral_vllm_generation.json",
```
The generation configuration parameters for all models and all prompting techniques that will be used in the experiment
```
"best_hyper_parameters_path": "/bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/experiment/configs/hyper-parameters-in-task.json",
```

The hyper-parameters for the experiment which should be specific to in-task, leave-one-out, and skill-transfer.
```
"output_dir": "/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/training_output",
```
The location for where the model will be stored
```
"tensorboard_logs": "/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/tensorboard",
```
The location for the tensorboard logs of the fine tuning experiment.
```
"argbench_dataset_path": "/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/argbench-dataset",
```
The location of the argbench experiment dataset. Notice that this the huggingface dataset and not the output of the benchmark preprocessing.

```
"model_configs_path": "/bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/experiment/configs/model_configs.json",
```

The location of the json file with all hugging face models (e.g., model names, prompt template, separator, and maximum allowed context size)  
### Jobs

4. To run the final in-task or cross- experiments you can run 
```
sbatch argbench/jobs/run_in_tasks_experiments.sh mistral-7b-inst-3 fallacy_detection_cmv_adhominem_habernal18
sbatch argbench/jobs/run_cross_tasks_experiments.sh mistral-7b-inst-3 fallacy_detection_cmv_adhominem_habernal18
```

5. To test a checkpoint on a specific dataset you can run 
```
sbatch argbench/jobs/run_cross_tasks_experiments.sh mistral-7b-inst-3 
```
 



