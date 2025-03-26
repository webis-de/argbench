# ArgBench


1. prepare the benchmark you have to run
```
sbatch argbench/jobs/preprocess_all_datasets.sh
```
2. Add your model to prompting or fine_tuning.json configuration

3. For fine-tuning, you can run the benchmark on one models
``` 
 sbatch argbench/jobs/fine_tuning_all_models.sh stance_classification_ibmsc_barhaim17
```
4. For prompting, you can run the benchmark on all datasets 


```
sbatch argbench/jobs/prompting.sh mistral-7b-inst-3
```
or for a specific dataset and a specific model

```
sbatch argbench/jobs/prompting.sh mistral-7b-inst-3 stance_classification_ibmsc_barhaim17
```

All jobs create results for the model in

/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/runs

and the time for the job can be tracked in 

/bigwork/nhwpajjy/task-specific-argument-mining-and-generation/argbench/jobs
