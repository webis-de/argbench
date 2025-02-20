import random
import pandas as pd
import json
import os
import numpy as np
from random import sample
from typing import List
from enum import Enum
from yaml import load, Loader
from pathlib import Path
import sys
import math
def read_tabular(path, separator=",", **kwargs):
    """Reads tabular csv file"""
    dataset = pd.read_csv(path, sep=separator, **kwargs)
    return dataset

def data_repo():
    """Get path of data repository"""
    curr_file = os.path.abspath(__file__)
    config_file = Path(curr_file).parents[2] / "config.yaml"
    with open(config_file, "r") as f:
        config = load(f, Loader=Loader)
        return Path(config["data_repo"])

def datasets_path():
    """Get path to dataset folder"""
    dataset_folder = data_repo() / "datasets"
    return dataset_folder

def tasks_path():
    """Path to tasks folder"""
    dataset_folder = data_repo() / "tasks"
    return dataset_folder

def add_seed_arg(argparser):
    """Adds seed argument to argument parser"""
    argparser.add_argument("-s", "--seed", default=42, help="Seed to use for dataset generation")

def set_seed(parsed_args):
    random.seed(parsed_args.seed)
    np.random.seed(parsed_args.seed)

def find_topic_size_to_split(df, topic_label):

    topics = df[topic_label].unique().tolist()

    df_test_ideal = df.sample(frac=0.2)
    test_topic_size = math.ceil(len(topics) * 0.2)

    min_dist = sys.maxsize
    best_test_topic = random.sample(topics, test_topic_size)
    for i in range(10):

        test_topics = random.sample(topics, test_topic_size)
        df_test = df[df[topic_label].isin(test_topics)]
        dist = abs(len(df_test) - len(df_test_ideal))
        if dist < min_dist:
            min_dist = dist
            best_test_topic = test_topics
    train_topics = [topic for topic in topics if topic not in best_test_topic]
    df_test = df[df[topic_label].isin(best_test_topic)]
    df_train = df[df[topic_label].isin(train_topics)]
    return df_test, df_train

def split_test_train(iteratable):
    size = len(list(iteratable))
    test_size = math.ceil(size * 0.2)
    test_indices = sample(list(range(size)), test_size)
    train_indices = [i for i in range(size) if i not in test_indices]
    test = [iteratable[i] for i in test_indices]
    train = [iteratable[i] for i in train_indices]
    return test, train
class Genres(Enum):
    """Valid genres"""
    ESSAYS = "essays"
    SOCIAL_MEDIA = "soical_media"
    WIKIPEDIA = "wikipedia"
    DEBATE_PORTALS = "debate_portals"
    DEBATES = "debates"
    NEWS = "news"
    WEB_FORUMS = "web-forums"
    WEB = "web"
    STORIES = "stories"


class Skills(Enum):
    """Valid subareas"""
    MINING = "mining" # *classification, segmentation, detection, extraction
    GENERATION = "generation" # *generation
    REASONING = "reasoning" # relation identification, similarity, key point matching
    QUALITY_ASSESSMENT = "quality-assessment" # *ranking
    PERSPECTIVE_ASSESSMENT = "perspective-assessment" # frame identification, controversy detection, warrant identification

class Tasks(Enum):
    """Valid tasks"""
    argument_canonicalization = "argument_canonicalization"
    argument_generation = "argument_generation"
    argument_ranking = "argument_ranking"
    argument_relation_identification = "argument_relation_identification"
    argument_similarity = "argument_similarity"
    argument_unit_classification = "argument_unit_classification"
    argument_unit_relation = "argument_unit_relation"
    argument_unit_segmentation = "argument_unit_segmentation"
    aspect_argument_generation = "aspect_argument_generation"
    aspect_detection = "aspect_detection"
    claim_improvement_detection = "claim_improvement_detection"
    conclusion_extraction = "conclusion_extraction"
    counter_argument_generation = "counter_argument_generation"
    fallacy_classification = "fallacy_classification"
    fallacy_detection = "fallacy_detection"
    frame_identification = "frame_identification"
    key_point_matching = "key_point_matching"
    post_controversy_detection = "post_controversy_detection"
    premise_extraction = "premise_extraction"
    premise_generation = "premise_generation"
    scheme_classification = "scheme_classification"
    stance_classification = "stance_classification"


class Output:
    """Output object"""

    def __init__(self, dataset_name) -> None:
        self.dataset_name = dataset_name
        self.tasks_path = tasks_path() / self.dataset_name
        self.contributors = []
        self.source = []
        self.url = []
        self.categories = []
        self.reasoning = []
        self.definition = []
        self.input_language = []
        self.output_language = []
        self.instruction_language = []
        self.domains = []
        self.positive_examples = []
        self.negative_examples = []
        self.instances = []
        self.genre = []
        self.instance_license = []
        self.tasks = []


    def append_positive_example(self, input: str, output: str, explanation: str):
        self.positive_examples.append({"input": input, "output": output, "explanation": explanation})


    def append_negative_example(self, input: str, output: str, explanation: str):
        self.negative_examples.append({"input": input, "output": output, "explanation": explanation})


    def append_instance(self, id: str, input: str, output: List[str]):
        self.instances.append({"id": id, "input": input, "output": output})


    def append_definition(self, definition: str):
        self.definition.append(definition)


    def append_subarea(self, subarea: Skills):
        assert subarea in Skills, f"Subarea {subarea} is not valid"
        self.categories.append(subarea.value)


    def append_genre(self, genre: Genres):
        assert genre in Genres, f"Genre {genre} is not valid"
        self.genre.append(genre.value)

    def append_task(self, task: Tasks):
        assert task in Tasks, f"Task {task} is not valid"
        self.tasks.append(task.value)


    def write_output(self, file_name):
        output = {
            "Contributors": self.contributors,
            "Source": self.source,
            "URL": self.url,
            "Categories": self.categories,
            "Reasoning": self.reasoning,
            "Definition": self.definition,
            "Input_language": self.input_language,
            "Output_language": self.output_language,
            "Instruction_language": self.instruction_language,
            "Domains": self.domains,
            "Positive Examples": self.positive_examples,
            "Negative Examples": self.negative_examples,
            "Instances": self.instances,
            "Genre": self.genre,
            "Instance License": self.instance_license
        }

        self.tasks_path.mkdir(parents=True, exist_ok=True)

        with open(self.tasks_path / file_name, "w+") as f:
            json.dump(output, f, indent=2)

class Metadata:
    """Metadata object for the dataset"""

    metadata_path = tasks_path() / "metadata.json"
    evaluation_metrics = [
        "fscore",
        "argument-bio-fscore",
        "fallacy-bio-fscore",
        "aspect-bio-fscore",
        "sentence-fscore",
        "bleu"
    ]

    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name
        if os.path.isfile(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                self.dataset_data = json.load(f)
        else:
            self.dataset_data = {}

        self.dataset_data[self.dataset_name] = {
            "file_list": [],
            "split_mapping": {},
            "evaluation_metrics": [],
            "skill": "",
            "genre": "",

        }


    def add_skill(self, skill: Skills):
        assert skill in Skills, f"Subarea {skill} is not valid"
        self.dataset_data[self.dataset_name]["skill"]= skill.value


    def add_genre(self, genre: Genres):
        assert genre in Genres, f"Genre {genre} is not valid"
        self.dataset_data[self.dataset_name]["genre"] = genre.value

    def append_task(self, task: Tasks):
        assert task in Tasks, f"Task {task} is not valid"
        self.dataset_data[self.dataset_name]["task"].append(task.value)

    def add_dataset(self, dataset_file, dataset_split=None):
        """Add Dataset to metadata"""
        dataset_split = dataset_split if dataset_split else "none"
        self.dataset_data[self.dataset_name]["file_list"].append(dataset_file)
        self.dataset_data[self.dataset_name]["split_mapping"][dataset_file] = dataset_split


    def add_evaluation_metric(self, metric):
        """Add metric to evaluate dataset on"""
        assert metric in self.evaluation_metrics, f"Metric should be one of: {self.evaluation_metrics}"
        self.dataset_data[self.dataset_name]["evaluation_metrics"].append(metric)


    def write_metadata(self):
        """Write metadata file to disk"""
        with open(self.metadata_path, "w") as f:
            json.dump(self.dataset_data, f, indent=2)

