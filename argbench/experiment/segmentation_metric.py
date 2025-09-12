import json
import logging
import string
from collections import defaultdict

import numpy as np
import regex as re
from nltk import word_tokenize
from sklearn.metrics import f1_score


logger = logging.getLogger()


def formulate_regex(labels):
    labels = "|".join(labels + [label.lower() for label in  labels])
    return f"(?:(?:{labels}):\s?[^\n]+\n+)*(?:(?:{labels}):\s?[^\n]+)"

def parse(text, labels):

    logger.debug(f"segmenting the document {text}")
    regular_expression = formulate_regex(labels)
    match  = re.search(regular_expression, text)
    parsed_document = defaultdict(list)
    if match:
        best_match = match.group(0)
        spans = best_match.split("\n")
        for span in spans:
            logger.debug(f"the sentence to be segmented {span}")
            if span and ":" in span:
                span_label = span.split(":")[0].strip().lower()
                span_text = span.split(":")[1].strip()
                regex_start_punc = "^[" + re.escape(string.punctuation) + "]+"
                regex_end_punc = "[" + re.escape(string.punctuation) + "]+$"
                clean_span_text= re.sub(regex_end_punc, "", span_text)
                clean_span_text= re.sub(regex_start_punc, "", clean_span_text)
                parsed_document[span_label].append(clean_span_text)
    return parsed_document

def compute_seg_match_f1_score(predictions, references,  labels, label_out):

    metric = {}

    for label in labels:
        label = label.lower()
        if label in label_out:
            continue
        tp = 0
        fp = 0
        fn = 0
        p = 0
        r = 0
        f1 = 0
        all_f1 = []
        for i, text in enumerate(references):
            reference = references[i].lower()
            prediction = predictions[i].lower()

            reference_spans = parse(reference, labels)
            prediction_spans = parse(prediction, labels)
            logger.debug(f"evaluating predictions {prediction_spans} and references {reference_spans}")
            if label in prediction_spans:
                for prediction_span in prediction_spans[label]:
                    if prediction_span in reference_spans[label]:
                        tp += 1
                    else:
                        fp += 1

            for reference_span in reference_spans[label]:
                if label not in prediction_spans or reference_span not in prediction_spans[label]:
                    fn += 1

        if tp + fp == 0:
            p = 0
        else:
            p = tp/(tp+fp)

        if tp + fn == 0:
            r = 0
        else:
            r = tp/(tp+fn)
        if p + r == 0:
            f1 = 0
        else:
            f1 = 2*p*r/(p+r)

        logger.debug(f"true positives {tp}, false positives fp {fp} false negatives {fn}")
        logger.debug(f"{label}-f1 {f1}")
        all_f1.append(f1)
        metric[f"{label}-precision"] = p
        metric[f"{label}-recall"] = r
        metric[f"{label}-fscore"] = f1
    metric["fscore"] = sum(all_f1)/len(all_f1)

    return metric






















def convert_to_bio(input, output, label_mappings):
    output_label = []

    for unit_idx, output_unit in enumerate(output):

        unit = list(output_unit.keys())[0]
        sentence_label = list(output_unit.values())[0]
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
        output_dict = json.loads(reference)
        ground_truth_dict = output_dict["output"]

        ground_truth_labels = convert_to_bio(input, ground_truth_dict, label_mappings)
        predictions_labels = convert_to_bio(input, prediction, label_mappings)
        #set_trace()
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
        "Ad-hominem" : "Fal-ad-hom",
        "Appeal to Emotion": "Fal-appeal-emotio",
        "Appeal to Authority": "Fal-app-author",
        "Slippery Slope": "Fal-slipper-slope",
        "False Cause": "Fal-false-cause",
        "Slogans": "Fal-slogans",
        "No-fallacy": "Fal-O"
    }

    return compute_bio_f1_score(predictions, references, inputs, map)
