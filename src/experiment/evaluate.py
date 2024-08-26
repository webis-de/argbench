#!/usr/bin/env python3
from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq, GenerationConfig
from peft import PeftModel, prepare_model_for_kbit_training
from preprocess import collect_datasets, tasks_path, get_metadata
from sklearn.metrics import f1_score
from argparse import ArgumentParser
import evaluate
import torch

def compute_f1_score(predictions, references, f1_average="macro"):
    labels = list(set(references))
    labels = sorted(labels, key=lambda r: len(r), reverse=True)
    label_mapping = {l: i for i, l in enumerate(labels)}
    label_targets = [label_mapping[r] for r in references]
    label_predictions = []
    for predicted in predictions:
        for label_name, label_idx in label_mapping.items():
            if label_name in predicted:
                label_predictions.append(label_idx)
    return f1_score(label_targets, label_predictions, average=f1_average)


def compute_rouge_score(predictions, references):
    rouge = evaluate.load("rouge")
    return rouge.compute(predictions=predictions, references=references)


def compute_bleu_score(predictions, references):
    bleu = evaluate.load("bleu")
    return bleu.compute(predictions=predictions, references=references)
