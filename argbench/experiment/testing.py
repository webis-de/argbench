#!/usr/bin/env python3
import evaluate
import re
import numpy as np
import logging

from IPython.core.debugger import set_trace
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_selection import r_regression
from sklearn.metrics import f1_score
from scipy.stats import kendalltau
from nltk import word_tokenize
from evaluate import load
from bert_score import score

logger = logging.getLogger(__name__)

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
#            labels=[label_mapping[l] for l in labels],
            beta=beta,
            average=f1_average
        )
    scores = {}
    for i, label in enumerate(labels):
        scores.update({f"{label}-precision": precision[i], f"{label}-recall": recall[i], f"{label}-fscore": fscore[i] })
    return scores




def compute_f1_score(predictions, references):
    logger.log(level=logging.INFO, msg=f"referecnes {references}")
    logger.log(level=logging.INFO, msg=f"predictions {predictions}")

    labels = set(references)
    labels_lowered = {label.lower().strip() for label in labels}
    mappings = {}
    counter = 0
    for label in labels_lowered:
        mappings[label] = counter
        counter = counter + 1
    predictions_int = []
    for i, prediction in enumerate(predictions):
        found = False
        if prediction.startswith("Output:"):
            prediction = prediction.replace("Output:", "")
        for label in labels_lowered:
            if prediction.strip().lower().startswith(label):
                predictions_int.append(mappings[label])
                found = True
                break
        if not found:
            predictions_int.append(-1)
    references_int = [mappings[reference.lower()] for reference in references]
    logger.log(level=logging.INFO, msg=f"referecnes {references_int}")
    logger.log(level=logging.INFO, msg=f"predictions {predictions_int}")

    score = f1_score(references_int, predictions_int, average = "macro")

    return {"fscore": score}

def compute_rouge_score(predictions, references):
    """
    Compute ROUGE scores for model predictions

    :param predictions: Model outputs
    :param references: True labels
    :returns: dict with different ROUGE scores
    """
    rouge = evaluate.load("rouge")
    rouge_score = rouge.compute(predictions=predictions, references=references)
    return {"rouge": rouge_score}

def compute_bleu_score(predictions, references):
    """
    Compute BLEU scores for model predictions

    :param predictions: Model outputs
    :param references: True labels
    :returns: dict with different BLEU scores
    """
    bleu = evaluate.load("bleu")
    bleu_score = bleu.compute(predictions=predictions, references=references)
    return {"bleu": bleu_score["bleu"]}

def compute_bert_score(predictions, references):

    P, R, F1 = score(predictions, references, lang='en')
    return {"bertscore": F1.mean().item()}


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
    kendall_score =  kendalltau(predictions, references)
    return {"kendall" : kendall_score}

def compute_meteor_score(predictions, references):
    meteor = evaluate.load("meteor")
    meteor_score = meteor.compute(predictions=predictions, references=references)
    return {"meteor" : meteor_score}


def convert_to_bio_text(input, output, label_mappings):
    output_label = []
    for unit_idx, unit in enumerate(output.split("\n")):
        if unit.strip():
            text = unit.strip()
            if ":" not in text:
                logger.log(level=logging.INFO, msg=f"argument unit segmentation eval error {text} does not contain a colon")
                output_label.extend([("Arg-O", token) for token in word_tokenize(text)])
                continue
            sentence_label = text.split(":")[0]
            unit  = text.split(":")[1]
            unit_tokens = word_tokenize(unit)

            for label in label_mappings:
                if sentence_label == label :
                    if sentence_label.startswith("No"):
                        token_label = label_mappings[sentence_label]
                        output_label.extend([(token_label, token) for token in unit_tokens])
                    else:
                        token_label = label_mappings[sentence_label]
                        output_label.append((f"{token_label}-B", unit_tokens[0]))
                        output_label.extend([(f"{token_label}-I", token) for token in unit_tokens[1:]])


    return output_label

def convert_to_bio(input, output, label_mappings):
    output_label = []

    for unit_idx, output_unit in enumerate(output):

        sentence_label = output_unit.keys()[1]
        unit  = output_unit.keys()[0]
        unit_tokens = word_tokenize(unit)

        for label in label_mappings:
            if sentence_label == label :
                if sentence_label.startswith("No"):
                    token_label = label_mappings[sentence_label]
                    output_label.extend([(token_label, token) for token in unit_tokens])
                else:
                    token_label = label_mappings[sentence_label]
                    output_label.append((f"{token_label}-B", unit_tokens[0]))
                    output_label.extend([(f"{token_label}-I", token) for token in unit_tokens[1:]])

    return output_label


def compute_bio_f1_score(predictions, references, inputs, label_mappings):
    all_labels = [ ]
    all_predictions = []
    #set_trace()
    labels = list(label_mappings.keys())
    for i, document in enumerate(inputs):
        prediction = predictions[i]
        reference = references[i]
        input = inputs[i]

        ground_truth_labels = convert_to_bio_text(input, reference, label_mappings)
        predictions_labels = convert_to_bio(input, prediction, label_mappings)
        if len(predictions_labels) < len(ground_truth_labels):
            for i in range(len(ground_truth_labels) - len(predictions_labels)):
                ground_truth_remaining = ground_truth_labels[len(predictions_labels):]
                predictions_labels.extend([(np.random.choice(labels), token) for (_, token) in ground_truth_remaining])
        else:
            predictions_labels = predictions_labels[:len(ground_truth_labels)]

        all_labels.extend([token[0] for token in ground_truth_labels])
        all_predictions.extend([token[0] for token in predictions_labels])


    #print(precision_recall_fscore_support(all_labels,all_predictions, average=None, labels=['Arg-I']))
    #print(f" Arg-B f1 {f1_score(all_labels, all_predictions, average=None, labels=['Arg-B'])}")
    #print(f" Arg-I f1 {f1_score(all_labels, all_predictions, average=None, labels=['Arg-I'])}")
    #print(f" Arg-O f1 {f1_score(all_labels, all_predictions, average=None, labels=['Arg-O'])}")
    macro_f1 = f1_score(all_labels, all_predictions, average='macro')
    token_labels = label_mappings.values()
    metrics = {"fscore": macro_f1}
    for token_label in token_labels:
        metrics[f"{token_label}_fscore"] = f1_score(all_labels, all_predictions, average=None, labels=[token_label])
    return metrics

def compute_seg_bio_f1_score (predictions, references, inputs):
    label_mappings = {"Argumentative": "Arg", "Non-argumentative":"Arg-O"}
    return compute_bio_f1_score(predictions, references, inputs, label_mappings)

def compute_aspect_bio_f1_score(predictions, references, inputs):
    label_mappings = {"Aspect": "Asp", "Not-aspect": "Asp-O"}
    return compute_bio_f1_score(predictions, references, inputs, label_mappings)

def compute_fallacy_bio_f1_score(predictions, references, inputs):
    map ={
         "Ad Hominem" : "Fal-ad-hom",
         "Appeal to Emotion": "Fal-appeal-emotio",
         "Appeal to Authority": "Fal-app-author",
         "Slippery Slope": "Fal-slipper-slope",
         "False Cause": "Fal-false-cause",
         "Slogans": "Fal-slogans",
         "No-fallacy": "Fal-O"
    }

    return compute_bio_f1_score(predictions, references, inputs, map)

def extract_sentence_labels(text):
    sentences = text.split("\n")
    labels = []
    for sentence in sentences:
        if sentence.strip():
            tokens = sentence.split(":")
            labels.append(tokens[0].strip().lower())
    return labels, sentences

def compute_sentence_f1(predictions, references, inputs):
    all_labels = [ ]
    all_predictions = []
    for i, document in enumerate(inputs):
        prediction = predictions[i]
        reference = references[i]
        input = inputs[i]

        ground_truth_labels, reference_sentences = extract_sentence_labels(reference)
        len_ground_truth_labels = len(ground_truth_labels)
        prediction_labels, prediction_sentences = extract_sentence_labels(prediction)
        if len(prediction_labels) > len_ground_truth_labels:
            prediction_labels = prediction_labels[:len_ground_truth_labels]
        elif len(prediction_labels) < len_ground_truth_labels:
            for i in range(len_ground_truth_labels - len(prediction_labels)):
                prediction_labels.append(np.random.choice(prediction_labels))
        logger.log(level=logging.INFO, msg=f"reference token size {len(reference.split())}")
        logger.log(level=logging.INFO, msg=f"prediction token size {len(prediction.split())}")
        #set_trace()
        logger.log(level=logging.INFO, msg=f"model did not output all needed documents for evaluation predicted sentences count ")
        logger.log(level=logging.INFO, msg=f"predicted sentences count {len(prediction_sentences)}")
        logger.log(level=logging.INFO, msg=f"reference sentences count {len(reference_sentences)}")
        logger.log(level=logging.INFO, msg=f"predicted labels {prediction_labels}")
        logger.log(level=logging.INFO, msg=f"ground labels {ground_truth_labels}")



        all_labels.extend(ground_truth_labels)
        all_predictions.extend(prediction_labels)
    if len(all_labels) == len(all_predictions):
        macro_f1 = f1_score(all_labels, all_predictions, average='macro')
    else:
        macro_f1 = 0
    logger.log(level=logging.INFO,msg=f"{macro_f1=}")
    return {"fscore" : macro_f1}
