# ArgBench

This repo comprises 32 datasets and 57 tasks covering 5 skills to evaluate how good are LLMs at computational argumentation tasks.
The skills are argument mining, argument perspective and quality assessment, argument reasoning, and argument generation.
You can evaluate your language model in Prompting, i.e. how good is your models at computational argumentation tasks. Or
how good is your model at generalizing to unseen computational tasks (Leave-one-out), by evaluating it on five hold-out tasks
after training it on the remaining tasks. 

To evaluate your models on the whole benchmark

### Prompting

1. Add your model to [argbench/experiment/configs/prompting/prompting.json](argbench/experiment/configs/prompting/prompting.json) configuration.
Each model should include three templates: zero-shot, few-shot, and Chain-of-thought prompting.


2. For prompting run the following script 

```
sbatch argbench/jobs/prompting.sh mistral-7b-inst-3
```
### Leave-one-out

3. For fine-tuning, add your model as described in prompting to [argbench/experiment/configs/fine-tuning/prompting.json](argbench/experiment/configs/fine-tuning/cross-task.json)
``` 
 sbatch argbench/jobs/cross_tasks.sh mistral-7b-inst-3
```

to your modely on one dataset you can run

**Prompting**
```
sbatch argbench/jobs/prompting_dataset.sh mistral-7b-inst-3 warrant_identification_semeval_2018_task_12_habernal18
```
**Leave-one-out**
```
sbatch argbench/jobs/cross_tasks.sh mistral-7b-inst-3 warrant_identification_semeval_2018_task_12_habernal18
```

All jobs create results for the model will be appended to the leader board located in the leaderboard path 
which can be configured in the configuration file
```
/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/runs
```
and the time for the job can be tracked in 
```
/bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/jobs
```
### Benchmark Preparation

1. To add your dataset to the benchmark, you should create a converter script in [argbench/experiment/converter](argbench/experiment/converter)
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

### Jobs

1. You can prompt all datasets with a prompting technique by one of the scripts

```
sbatch argbench/jobs/prompting_all_{1_shot,4_shot,cot}.sh mistral-7b-inst-3
```

2. You can prompt one dataset with a prompting technique by one of the scripts
```
sbatch argbench/jobs/prompting_datsaet_{1_shot,4_shot,cot}.sh fallacy_detection_cmv_adhominem_habernal18 mistral-7b-inst-3
```

3. To optimize hyper-prameters for a specific task or cross-task , you can run 

```
sbatch argbench/jobs/run_in_tasks_hpo.sh mistral-7b-inst-3 fallacy_detection_cmv_adhominem_habernal18
sbatch argbench/jobs/run_cross_tasks_hpo.sh mistral-7b-inst-3
```
4. To run the final in-task or cross- experiments you can run 
```
sbatch argbench/jobs/run_in_tasks_experiments.sh mistral-7b-inst-3 fallacy_detection_cmv_adhominem_habernal18
sbatch argbench/jobs/run_cross_tasks_experiments.sh mistral-7b-inst-3 fallacy_detection_cmv_adhominem_habernal18
```

5. To test a checkpoint on a specific dataset you can run 
```
sbatch argbench/jobs/run_cross_tasks_experiments.sh mistral-7b-inst-3 
```
 
6. To run all prompting experiments for a specific skill
```
sbatch argbench/jobs/run_prompting_skill_model.sh mining  mistral-7b-inst-3
```
7. To run all prompting experiments for a specific skill
```
./run_prompting_skill_model.sh mining  mistral-7b-inst-3
```
8. To run all in-task experiment son all models
```
./run_in_tasks_experiments_all_models.sh
```


