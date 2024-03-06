import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from transformers import (AutoModelForCausalLM, AutoTokenizer, BertForSequenceClassification, BertTokenizer, BertModel,
 RobertaForSequenceClassification, RobertaTokenizer, RobertaModel, TrainingArguments, Trainer)


from datasets import load_dataset, load_from_disk, Dataset, DatasetDict, concatenate_datasets
import evaluate
import wandb
import os
import argparse

from detector import LLMDetector

os.environ["WANDB_PROJECT"] = "gen_detector"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints


def tokenize_text(x, tokenizer):
    return tokenizer(x["text"], truncation=True, padding="max_length")

def run(num_epochs, model, dataset, learning_rate, warmup_ratio, weight_decay, batch_size):


    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        logging_steps=5,
        logging_dir="./logs",
        report_to="wandb",


    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--detector", type=str, help="Name of the model to train from: roberta, bert, e5", default="bert-base-uncased")
    parser.add_argument("--dataset_path", type=str, help="Path to the fake true dataset (generated with generate_fake_true_dataset.py)", default="fake_true_dataset")
    parser.add_argument("--batch_size", type=int, help="Batch size to train the model", default=8)
    parser.add_argument("--num_epochs", type=int, help="Number of epochs to train the model", default=3)
    parser.add_argument("--learning_rate", type=float, help="Learning rate for the model", default=5e-3)
    parser.add_argument("--warmup_ratio", type=float, help="Warmup ratio for the model", default=0.1)
    parser.add_argument("--weight_decay", type=float, help="Weight decay for the model", default=0.01)
    parser.add_argument("--device", type=str, help="Device to train the model", default="cuda")
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset_path)


    if args.detector == "roberta":
        detector_path = "openai-community/roberta-base-openai-detector"
        detector_model = RobertaForSequenceClassification.from_pretrained(detector_path).to(args.device)
        bert_tokenizer = RobertaTokenizer.from_pretrained(detector_path)
        detector = LLMDetector(detector_model, bert_tokenizer, 2)
    else:
        raise ValueError("No other detector currently supported")
    

    # tokenize text
    dataset = dataset.map(lambda x: tokenize_text(x, bert_tokenizer), batched=True)

    run(args.num_epochs, detector, dataset, dataset, args.learning_rate, args.warmup_ratio, args.weight_decay, args.batch_size)


    