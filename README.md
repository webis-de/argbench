# ArgBench

This repo comprises 32 datasets and 22 tasks to evaluate how good are LLMs at computational argumentation tasks. 
You can evaluate your models in Prompting, i.e. how good is your models at computational argumentation tasks. Or
how good is your model at generalizing to unseen computational tasks (Leave-one-out-experiment).

To evaluate your models on the whole benchmark

### Prompting

1. Add your model to (experiment/configs/prompting/prompting.json)[experiment/configs/prompting/prompting.json] configuration.
Each model should include three templates: zero-shot, few-shot, and Chain-of-thought prompting.


2. For prompting run the following script 

```
sbatch argbench/jobs/prompting.sh mistral-7b-inst-3
```
### Fine-tuning 

3. For fine-tuning, add your model as described in prompting to (experiment/configs/fine-tuning/prompting.json)[experiment/configs/fine-tuning/cross-task.json]
``` 
 sbatch argbench/jobs/cross_tasks.sh mistral-7b-inst-3
```

to your modely on one dataset you can run


```
sbatch argbench/jobs/prompting_dataset.sh mistral-7b-inst-3 stance_classification_ibmsc_barhaim17
```

All jobs create results for the model will be appended to the leader board located in the leaderboard path 
which can be configured in the configuration file

/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/runs

and the time for the job can be tracked in 

/bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/jobs

### Benchmark Preparation

1. To add your dataset to the benchmark, you should create a converter script in (argbench/experiment/converter)[argbench/experiment/converter]
The script should include a task definition and how each instance should be converted to completion tasks. For each dataset,
you should add a training, test, and validation set. 

2. You have also to add the following
*Metadata*: 

- Skill (whether it is mining, quality assessment, perspective assessment, reasoning, or generation)
- The genre (e.g., Scoial Media)
- Evaluation metric (e.g., F1-score or BertScore)
- The file and their corresponding split, i.e., which part of the dataset is a training or test set.
3. To generate or prepare the benchmark you have to run
```
sbatch argbench/jobs/preprocess_all_datasets.sh
```

Push to github
 ```
 git push upstream main
 ```