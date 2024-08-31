#!/usr/bin/env python3
from sklearn.metrics import precision_recall_fscore_support
import evaluate

def compute_precision_recall_fscore_support(predictions, references, f1_average="macro", beta=1.0):
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
    rouge = evaluate.load("rouge")
    return rouge.compute(predictions=predictions, references=references)


def compute_bleu_score(predictions, references):
    bleu = evaluate.load("bleu")
    return bleu.compute(predictions=predictions, references=references)
