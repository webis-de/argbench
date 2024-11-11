#!/usr/bin/env python3
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_selection import r_regression
from sklearn.metrics import f1_score
from scipy.stats import kendalltau
from nltk import word_tokenize
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



def convert_to_bio(input, output):
    labels = []
    input_tokens = word_tokenize(input)
    last_argument_index = 0
    for i, input_token in enumerate(input_tokens):
        for argument in output.split("\n"):
            argument_tokens = word_tokenize(argument)
            if argument_tokens[0] == input_token and " ".join(argument_tokens) == " ".join(input_tokens[i:i+len(argument_tokens)]):
                labels.extend(["Arg-O" for _ in range(last_argument_index,i)])
                labels.append("Arg-B")
                labels.extend(["Arg-I" for _ in range(i+1, i+len(argument_tokens))])
                last_argument_index = i + len(argument_tokens) + 1
    if last_argument_index <= len(input_tokens):
        labels.extend(["Arg-O" for _ in range(last_argument_index,len(input_tokens)+1)])
    return labels

def compute_segmentation_f1_score(predictions, references, inputs):
    all_labels = [ ]
    all_predictions = []
    for i, document in enumerate(inputs):
        prediction = predictions[i]
        reference = references[i]
        input = inputs[i]
        groudn_truth_labels = convert_to_bio(input, reference)
        predictions_labels = convert_to_bio(input, prediction)
        all_labels.extend(groudn_truth_labels)
        all_predictions.extend(predictions_labels)
    #print(precision_recall_fscore_support(all_labels,all_predictions, average=None, labels=['Arg-I']))
    #print(f" Arg-B f1 {f1_score(all_labels, all_predictions, average=None, labels=['Arg-B'])}")
    #print(f" Arg-I f1 {f1_score(all_labels, all_predictions, average=None, labels=['Arg-I'])}")
    #print(f" Arg-O f1 {f1_score(all_labels, all_predictions, average=None, labels=['Arg-O'])}")
    macro_f1 = f1_score(all_labels, all_predictions, average='macro')
    argb_f1 = f1_score(all_labels, all_predictions, average=None, labels=['Arg-B'])
    argi_f1 = f1_score(all_labels, all_predictions, average=None, labels=['Arg-I'])
    argo_f1 = f1_score(all_labels, all_predictions, average=None, labels=['Arg-O'])
    metrics = {"fscore": macro_f1, "argb-fscore" : argb_f1, "argo-fscore": argo_f1, "argi-fscore": argi_f1}
    return metrics

