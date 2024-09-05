from preprocess import collect_datasets, tasks_path, get_metadata
from sklearn.metrics import f1_score
from collections import Counter
from pathlib import Path
from argparse import ArgumentParser
from evaluate import compute_precision_recall_fscore_support
from config import RunConfig
import random

if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Run peft finetuning experiment")

    arg_parser.add_argument("-c", "--config", type=Path, help="Path to experiment config")

    args = arg_parser.parse_args()

    config = RunConfig.from_file(args.config)

    datasets_path = tasks_path()
    metadata = get_metadata()
    _, test_instances = collect_datasets(
        config.train_datasets,
        config.test_datasets,
        metadata,
        datasets_path,
        True
    )

    unique_labels = list(set([data["output"] for data in test_instances]))

    predictions = [random.choice(unique_labels) for _ in range(len(test_instances))]

    labels = [d["output"] for d in test_instances]

    precision, recall, fscore, support, labels = compute_precision_recall_fscore_support(
        predictions,
        labels,
        f1_average=config.validation_config.fscore_average,
        beta=config.validation_config.fscore_beta
    )

    print(f"Precision: {precision} Recall: {recall} Fscore: {fscore} Support: {support} Labels: {labels}")
