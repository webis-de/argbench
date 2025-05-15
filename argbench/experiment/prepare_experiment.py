from collections import defaultdict
from typing import Dict, Set

import datasets
from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets

from argbench.converter.common import *
from argbench.experiment.preprocess import *
from argbench.experiment.utils import *

logger = logging.getLogger()


def get_dataset_split(dataset, set, metadata, task_data_path):

    for file in metadata[dataset]["file_list"]:
        if metadata[dataset]["split_mapping"][file] == set:

            return task_data_path / dataset / file
    return None

def load_set(dataset, task_data_path, split, sample_rate:float = None, sample_size: int = None):
    metadata = get_metadata()

    if split == DatasetSplit.TRAIN_AND_VAL:
        train_set,_ = load_set(dataset, task_data_path, DatasetSplit.TRAIN, sample_rate, sample_size)
        test_set,_ = load_set(dataset, task_data_path, DatasetSplit.VAL, sample_rate, sample_size)
        return pd.concat([train_set, test_set]), None
    split = split.value
    path = get_dataset_split(dataset, split, metadata, task_data_path)
    if sample_rate or sample_size:
        path_sample = formulate_sample_path(path, sample_rate, sample_size)
    else:
        path_sample = path
    logger.debug(f"path:{path}")
    logger.debug(f"sample path:{path_sample}")
    if os.path.exists(path_sample):
        return  pd.read_json(path_sample, lines=True), path_sample
    else:
        return sample_set(path, sample_rate, sample_size)

def formulate_sample_path(output_path: Path, sample_rate: float = None, sample_size: int= None):
    base, file_name = os.path.split(output_path)
    if sample_rate:
        sample_name = file_name.replace(".json", f"-rate-{sample_rate}.json")
    elif sample_size:
        sample_name = file_name.replace(".json", f"-size-{sample_size}.json")
    else:
        raise ValueError("no rate or size defined")

    return os.path.join(base, sample_name)

def sample_set(output_path: Path, sample_rate: float = None, sample_size: int = None ):

    df_set = pd.read_json(output_path, lines=True)
    path_sample = formulate_sample_path(output_path, sample_rate, sample_size)
    if sample_rate:
        df_sample = df_set.sample(frac=sample_rate)
    elif sample_size:
        df_sample = df_set.sample(sample_size)
    else:
        raise ValueError("no rate or size defined")
    df_sample.to_json(path_sample, orient='records', lines=True)
    return df_sample, path_sample


def create_dataset_in_tasks(task_data_path, prompt_technique_template, experiment_splits, test_subsample_rate=None, train_subsample_rate=None):

    def template_formatter(row):
        return prompt_technique_template.format(instance_input=row["document"], definition=row["definition"])

    ### if in-task iterate over test sets and create 5 datasets
    ## if cross-task iterate over test sets and create 5 datasets
    ## if cross-task iterate over validation sets and create 5 datasets
    ### if cross-task iterate over training set and create the rest datasets
    ### save the datasets to hugging face
    in_task_dataset = {}

    for task in experiment_splits["test"]:
        print(task)
        df_training, train_path = load_set(task, task_data_path, DatasetSplit.TRAIN, sample_rate=train_subsample_rate)
        df_test, test_path = load_set(task, task_data_path, DatasetSplit.TEST,  sample_rate=test_subsample_rate)
        df_validation, val_path = load_set(task, task_data_path, DatasetSplit.VAL, sample_rate=test_subsample_rate)
        dataframes = (df_training, df_test, df_validation)
        for df_split in dataframes:
            for column in df_split.columns:
                df_split[column] = df_split[column].astype(str)


            df_split.rename(columns={"input": "document"}, inplace=True)
            df_split["input"] = df_split.apply(template_formatter, axis=1)

        df_test = df_test[["id", "input", "output"]]
        df_training = df_training[["id", "input", "output"]]
        df_validation = df_validation[["id", "input", "output"]]

        hf_test = datasets.Dataset.from_pandas(df_test)
        hf_training = datasets.Dataset.from_pandas(df_training)
        hf_validation = datasets.Dataset.from_pandas(df_validation)

        df_training.path = train_path
        df_test.path = test_path
        df_validation.path = val_path


        dataset =  {f"test_{task}":hf_test, f"train_{task}":hf_training, f"val_{task}": hf_validation}
        in_task_dataset.update(dataset)
    hf_dataset = DatasetDict(in_task_dataset)
    return hf_dataset


def create_dataset_cross_tasks(task_data_path, prompt_technique_template, experiment_splits, test_subsample_rate=None, train_subsample_rate=None):
    leave_one_task_dataset = {}
    ### add eac validation set of each validation task as validation
    ### Iterate over each task in the test dataset and take its teset as a  test dataset
    ### for the task, we will run the validation on the validation tasks. For testing, we will take one random task as validation set
    ###
    def template_formatter(row):
        return prompt_technique_template.format(instance_input=row["document"], definition=row["definition"])

    for task in experiment_splits["test"]:


        df_test, test_path = load_set(task, task_data_path, DatasetSplit.TEST,  sample_rate=test_subsample_rate)

        for column in df_test.columns:
            df_test[column] = df_test[column].astype(str)
        df_test.rename(columns={"input": "document"}, inplace=True)
        df_test["input"] = df_test.apply(template_formatter, axis=1)

        df_test = df_test [ ["id", "input", "output"]]
        hf_test = datasets.Dataset.from_pandas(df_test)
        dataset = {f"test_{task}":hf_test}
        leave_one_task_dataset.update(dataset)



    for task in experiment_splits["validation"]:
        print(task)
        df_validation, validation_path = load_set(task, task_data_path, DatasetSplit.VAL, sample_rate=test_subsample_rate)
        for column in df_validation.columns:
            df_validation[column] = df_validation[column].astype(str)
        df_validation.rename(columns={"input": "document"}, inplace=True)
        df_validation["input"] = df_validation.apply(template_formatter, axis=1)

        df_validation = df_validation [ ["id", "input", "output"]]

        hf_validation = datasets.Dataset.from_pandas(df_validation)
        dataset = {f"val_{task}":hf_validation}
        leave_one_task_dataset.update(dataset)


    for task in experiment_splits["training"]:
        df_training, training_path = load_set(task, task_data_path, DatasetSplit.VAL, sample_rate=train_subsample_rate)
        for column in df_training.columns:
            df_training[column] = df_training[column].astype(str)
        df_training.rename(columns={"input": "document"}, inplace=True)
        df_training["input"] = df_training.apply(template_formatter, axis=1)

        df_training = df_training[["id", "input", "output"]]
        hf_training = datasets.Dataset.from_pandas(df_training)
        leave_one_task_dataset.update({f"train_{task}": hf_training})

    hf_leave_one_task_dataset = DatasetDict(leave_one_task_dataset)
    return hf_leave_one_task_dataset

def create_dataset_prompting(task_data_path, prompting_technique_template, test_subsample_rate=None, few_shot_amount=None):

    def few_shot_template_formatter(row):
        return prompting_technique_template.format(instance_input=row["document"], definition=row["definition"], example=row["example"])

    def template_formatter(row):
        return prompting_technique_template.format(instance_input=row["document"], definition=row["definition"])

    prompting_datasets = {}

    tasks = get_metadata()

    for task in tasks:
        if few_shot_amount:
            df_training, train_path = load_set(task, task_data_path, DatasetSplit.TRAIN_AND_VAL, sample_size=few_shot_amount)
            df_training.path = train_path
            for column in df_training.columns:
                df_training[column] = df_training[column].astype(str)
            df_training = df_training[["id", "input", "output"]]



#        df_test['output'] = df_test['output'].apply(json.dumps)
        df_test, test_path = load_set(task, task_data_path, DatasetSplit.TEST,sample_rate = test_subsample_rate)

        df_test.path = test_path

        ###  Formatting
        if few_shot_amount:
            example_str = ""
            counter = 0
            for _, example in df_training.iterrows():
                example_instance = f'\nExample {counter}:\n Input: {example["input"]}\nOutput: {example["output"]}\n'
                example_str += example_instance
                counter += 1
            df_test.rename(columns={"input": "document"}, inplace=True)
            df_test["example"] = example_str
            df_test["input"] = df_test.apply(few_shot_template_formatter, axis=1)
        else:
            df_test.rename(columns={"input": "document"}, inplace=True)
            df_test["input"] = df_test.apply(template_formatter, axis=1)

        for column in df_test.columns:
            df_test[column] = df_test[column].astype(str)

        df_test = df_test[["id","document", "input", "output"]]
        prompting_datasets[f"test_{task}"]= Dataset.from_pandas(df_test)
    hf_dataset_prompting = DatasetDict(prompting_datasets)

    return hf_dataset_prompting

    ### TODO: create datasets using huggingface api

def formulate_argbench_dataset_path(experiment_type:ExperimentType, prompting_technique: PromptingTechnique, sample:bool, path_argbench_dataset: Path) -> Path:
    if experiment_type == ExperimentType.PROMPTING:
        dataset = f"argbench-{experiment_type.value}-{prompting_technique.value}"
    else:
        dataset = f"argbench-{experiment_type.value}"
    if sample:
        dataset = dataset + "-small"
    return path_argbench_dataset / dataset

def get_cot_prompt_template():
    return "{definition}\nThink step by step and prepend your output with Output:\n{instance_input}"

def get_shot_prompt_template():
    return "{definition}\nDo not explain and do not rephrase the input.\n###Examples:\n{example}\n###\n{instance_input}"

def get_zero_shot_prompt_template():
    return "{definition}\nDo not explain and do not rephrase the input.\n{instance_input}"
def create_argbench_dataset(experiment_type: ExperimentType, prompting_technique: PromptingTechnique, sample: bool, run_config: RunConfig)-> DatasetDict:
    """
    Use RunConfig to create train and test datasets

    :param run_config: RunConfig with train_datasets and test_datasets config dicts
    :returns: Tuple of train and test datasets in pandas DataFrame
    """
    ### TODO add the configuration to path
    path_argbench_dataset = Path(run_config.argbench_dataset_path)
    tasks_path = Path(run_config.data_folder)

    path_dataset = formulate_argbench_dataset_path(experiment_type, prompting_technique, sample, path_argbench_dataset)
    # path_dataset = formulate_argbench_dataset_path(ExperimentType.PROMPTING, PromptingTechnique.ZERO_SHOT, sample=sample, path_argbench_dataset=path_argbench_dataset)
    # path_four_shot_dataset = formulate_argbench_dataset_path(ExperimentType.PROMPTING, PromptingTechnique.FOUR_SHOT, sample=sample, path_argbench_dataset=path_argbench_dataset)
    # path_one_shot_dataset = formulate_argbench_dataset_path(ExperimentType.PROMPTING, PromptingTechnique.ONE_SHOT, sample=sample, path_argbench_dataset=path_argbench_dataset)
    # path_cot_dataset = formulate_argbench_dataset_path(ExperimentType.PROMPTING, PromptingTechnique.COT, sample=sample, path_argbench_dataset=path_argbench_dataset)
    # path_in_task_dataset = formulate_argbench_dataset_path(ExperimentType.IN_TASK, PromptingTechnique.ZERO_SHOT, sample=sample, path_argbench_dataset=path_argbench_dataset)
    if prompting_technique == PromptingTechnique.ZERO_SHOT:
        prompt_template = get_zero_shot_prompt_template()
    elif prompting_technique == PromptingTechnique.ONE_SHOT or prompting_technique == PromptingTechnique.FOUR_SHOT:
        prompt_template = get_shot_prompt_template()
    else:
        prompt_template = get_cot_prompt_template()

    if prompting_technique == PromptingTechnique.FOUR_SHOT:
        few_shot_count = 4
    elif prompting_technique == PromptingTechnique.FOUR_SHOT:
        few_shot_count = 1
    else:
        few_shot_count = None

    with open(run_config.experiment_splits_path) as experiment_splits_file:
        experiment_splits = json.load(experiment_splits_file)
    if experiment_type == ExperimentType.IN_TASK:
        if sample:
            dataset = create_dataset_in_tasks(tasks_path, prompt_template, experiment_splits, 0.1, 0.5)
        else:
            dataset = create_dataset_in_tasks(tasks_path, prompt_template, experiment_splits)
    elif experiment_type == ExperimentType.PROMPTING:
            if sample:
                dataset = create_dataset_prompting(tasks_path, prompt_template,0.1, few_shot_count )
            else:
                dataset = create_dataset_prompting(tasks_path, prompt_template, test_subsample_rate=None, few_shot_amount=few_shot_count )
    elif experiment_type == ExperimentType.LEAVE_ONE_TASK:
        if sample:
            dataset = create_dataset_cross_tasks(tasks_path, prompt_template, experiment_splits, 0.1, 0.5)
        else:
            dataset = create_dataset_cross_tasks(tasks_path, prompt_template, experiment_splits)
    else:
        ### TODO
        pass
    dataset.save_to_disk(path_dataset)
    return dataset

def load_experiment(experiment_type, prompting_technique, sample, test_task, run_config: RunConfig):
    path_argbench_dataset = Path(run_config.argbench_dataset_path)
    path_dataset = formulate_argbench_dataset_path(experiment_type, prompting_technique, sample, path_argbench_dataset)

    val_split = f"val_{test_task}"
    train_split =  f"train_{test_task}"
    test_split = f"test_{test_task}"

    if path_dataset.exists():
        dataset = load_from_disk(path_dataset)
    else:
        dataset = create_argbench_dataset(experiment_type, prompting_technique, sample, run_config)
    if experiment_type == ExperimentType.IN_TASK:
        val_dataset =  dataset.pop(val_split)
        train_dataset = dataset.pop(train_split)
        test_dataset = dataset.pop(test_split)
        dataset = DatasetDict({"train": train_dataset, "val": val_dataset, "test": test_dataset})
        return dataset
    elif experiment_type == ExperimentType.PROMPTING:
        if not test_task:
            return dataset
        else:
            task_dataset = dataset.pop(f"test_{test_task}")
            return DatasetDict({f"test_{test_task}":task_dataset})
    elif experiment_type == ExperimentType.LEAVE_ONE_TASK:
        validation_tasks = [task.replace("val_","") for task in dataset.keys() if "val" in task]
        if test_task in validation_tasks: ## Validation Experiment

            val_split = f"val_{test_task}"
            training_datasets = [dataset[split] for split in dataset.keys() if split != val_split]
            val_dataset = dataset.pop(val_split)
            train_dataset = concatenate_datasets(training_datasets)
            dataset = DatasetDict({"train": train_dataset, "val": val_dataset, "test": val_dataset})
        else:                             ## Test Experiment
            val_split = "val_stance_classification_ukp_sentential_stab18" ## Fixed this for validation
            training_datasets = [dataset[split] for split in dataset.keys() if split != test_task and split !=val_split]
            train_dataset = concatenate_datasets(training_datasets)
            test_dataset = dataset.pop(test_split)
            val_dataset = dataset.pop(val_split)
            dataset = DatasetDict({"train": train_dataset, "val": val_dataset, "test": test_dataset})

        return dataset
    else:
        return None


def get_filters_by_skill(skill: str) -> Dict[str, float]:
    filter = {}
    metadata = get_metadata()
    for task in metadata:
        if metadata[task]["skill"] == skill:
            filter[task] = 1
    return filter


def get_experiment_split(is_validate, experiment_splits) -> (Set[str], Set[str]):
    """

    :param is_validate: whether the experiment is a validation or test experiment
    :param experiment_splits: a dictionary containing
    :return:
    """
    if is_validate:
        test = experiment_splits["validation"]
        training = experiment_splits["training"]
    else:
        test = experiment_splits["test"]
        training = experiment_splits["validation"] + experiment_splits["training"]
    return test, training


def save_experiment_splits(experiment_splits_path):
    """
    This function randomly choose five tasks for validation, five for test, and the rest for training
    :param run_config: the main configuration object
    :return: None
    """


    metadata = get_metadata()
    tasks_per_skill = defaultdict(list)
    experiment_splits = defaultdict(list)

    for task in metadata:
        skill = metadata[task]["skill"]
        tasks_per_skill[skill].append(task)

    for skill in tasks_per_skill:
        val_task = np.random.choice(tasks_per_skill[skill])
        test_task = np.random.choice(tasks_per_skill[skill])
        while test_task == val_task:
            test_task = np.random.choice(tasks_per_skill[skill])

        experiment_splits["validation"].append(val_task)
        experiment_splits["test"].append(test_task)

    for task in metadata:
        if task not in experiment_splits["test"] and task not in experiment_splits["validation"]:
            experiment_splits["training"].append(task)

    with open(experiment_splits_path, "w") as file_stream:
        json.dump(experiment_splits, file_stream, indent=4, sort_keys=True)






