#!/usr/bin/env python3
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_selection import r_regression
from scipy.stats import kendalltau
import evaluate
import re
import numpy as np

def compute_precision_recall_fscore_support(predictions, references, f1_average="macro", beta=1.0):
    """
    Perform F1 evaluation for model outputs

    :param predictions: Model outputs
    :param references: True labels
    :param f1_average: Averaging method for f1 score
    :param beta: Beta weight of f1
    :returns: Tuple of precision, recall, f1 score, support and all labels
    """
    labels = list(set(references))
    labels = sorted(labels, key=lambda r: len(r), reverse=True)
    label_mapping = {l: i for i, l in enumerate(labels)}
    label_targets = [label_mapping[r] for r in references]
    label_predictions = []
    for predicted in predictions:
        is_label_not_found = True
        for label_name, label_idx in label_mapping.items():
            if label_name in predicted:
                label_predictions.append(label_idx)
                is_label_not_found = False
                break
        if is_label_not_found:
            new_label = max(idx for idx in label_mapping.values()) + 1
            label_mapping[predicted] = new_label
            label_predictions.append(new_label)

    precision, recall, fscore, support = precision_recall_fscore_support(
            label_targets,
            label_predictions,
            labels=[label_mapping[l] for l in labels],
            beta=beta,
            average=f1_average
        )
    return (
        precision,
        recall,
        fscore,
        support,
        labels
    )


def compute_rouge_score(predictions, references):
    """
    Compute ROUGE scores for model predictions

    :param predictions: Model outputs
    :param references: True labels
    :returns: dict with different ROUGE scores
    """
    rouge = evaluate.load("rouge")
    return rouge.compute(predictions=predictions, references=references)


def compute_bleu_score(predictions, references):
    """
    Compute BLEU scores for model predictions

    :param predictions: Model outputs
    :param references: True labels
    :returns: dict with different BLEU scores
    """
    bleu = evaluate.load("bleu")
    return bleu.compute(predictions=predictions, references=references)


def rank_string_to_matrix(rank_strings):
    """
    Convert string rankings to numpy matrix

    :param rank_strings: List of strings with ranking
    :returns: Numpy matrix with rankings
    """
    rank_regex = re.compile("\d+")
    prediction_ranks = []
    for p in rank_strings:
        rr = rank_regex.findall(p)
        prediction_ranks.append(rr)

    return np.array(prediction_ranks)


def compute_kendall_tau(predictions, references):
    """
    Compute kendall tau metric

    :param predictions: Model outputs
    :param references: True labels
    :returns: dict with different BLEU scores
    """
    predictions = rank_string_to_matrix(predictions)
    references = rank_string_to_matrix(references)
    return kendalltau(predictions, references)

def compute_meteor_score(predictions, references):
    rouge = evaluate.load("meteor")
    return rouge.compute(predictions=predictions, references=references)