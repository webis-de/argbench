from argparse import ArgumentParser
from pathlib import Path
from collections import Counter
from preprocess import collect_datasets, tasks_path, get_metadata
from evaluate import compute_precision_recall_fscore_support
from config import RunConfig


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Run peft finetuning experiment")

    arg_parser.add_argument("-c", "--config", type=Path, action="append", help="Path to experiment config")

    RunConfig.register_cli(arg_parser)

    args = arg_parser.parse_args()

    config = RunConfig.from_file(args.config, args)

    config.is_eval = True

    datasets_path = tasks_path()
    metadata = get_metadata()
    _, test_instances = collect_datasets(config)

    predictions = test_instances["output"].value_counts().reset_index()

    most_common_label = predictions["output"].iloc[0]
    predictions = [most_common_label] * len(test_instances)
    labels = test_instances["output"]

    precision, recall, fscore, support, labels = compute_precision_recall_fscore_support(
        predictions,
        labels,
        f1_average=config.validation_config.fscore_average,
        beta=config.validation_config.fscore_beta
    )

    print(f"Precision: {precision} Recall: {recall} Fscore: {fscore} Support: {support} Labels: {labels}")
