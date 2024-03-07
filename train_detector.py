import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from transformers import (AutoModelForCausalLM, AutoTokenizer, BertForSequenceClassification, BertTokenizer, BertModel,
 RobertaForSequenceClassification, RobertaTokenizer, RobertaModel, TrainingArguments, Trainer, DataCollatorWithPadding,
    TrainerCallback)
from copy import deepcopy


from datasets import load_dataset, load_from_disk, Dataset, DatasetDict, concatenate_datasets
import evaluate
import wandb
import os
import argparse

from detector import LLMDetector



def tokenize_text(x, tokenizer):
    return tokenizer(x["text"], truncation=True, padding="max_length")

def run(num_epochs, model, tokenizer, dataset, learning_rate, warmup_ratio, weight_decay, batch_size):


    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        print("labels", labels)
        print("predictions", predictions)
        return metric.compute(predictions=predictions, references=labels)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        #output_dir="./results",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        logging_steps=50,
        eval_steps=100,
        # This is important to set evaluation strategy, otherwise there will be no evaluation
        evaluation_strategy="steps",
        save_steps=200,
        save_total_limit=4,
        #logging_dir="./logs",
        #report_to="wandb",
        seed=42,
        fp16=True,
    )

    """
    class CustomCallback(TrainerCallback):
        
        def __init__(self, trainer) -> None:
            super().__init__()
            self._trainer = trainer
        
        def on_evaluate(self, args, state, control, **kwargs):
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy
    """
        

    
    """
    class CustomTrainer(Trainer):

        def compute_loss(self, model, inputs, return_outputs=False):
            if self.label_smoother is not None and "labels" in inputs:
                labels = inputs.pop("labels")
            else:
                labels = None
            outputs = model(**inputs)

            # code for calculating accuracy
            if "labels" in inputs:
                logits = outputs.logits.detach()
                preds = logits.argmax(dim=1)
                acc1 = metric.compute(predictions=preds, references=inputs.labels)
                self.log({'accuracy_score': acc1})
                acc = (
                    (preds.argmax(axis=-1) == inputs.labels.reshape(1, len(inputs.labels))[0])
                    .type(torch.float)
                    .mean()
                    .item()
                )
                self.log({"train_accuracy": acc})
            # end code for calculating accuracy
                        
            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            if labels is not None:
                loss = self.label_smoother(outputs, labels)
            else:
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            return (loss, outputs) if return_outputs else loss
        
    """

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )
    #trainer.add_callback(CustomCallback(trainer))
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--detector", type=str, help="Name of the model to train from: roberta, bert, e5", default="roberta")
    parser.add_argument("--dataset_path", type=str, help="Path to the fake true dataset (generated with generate_fake_true_dataset.py)", default="fake_true_dataset")
    parser.add_argument("--batch_size", type=int, help="Batch size to train the model", default=8)
    parser.add_argument("--num_epochs", type=int, help="Number of epochs to train the model", default=3)
    parser.add_argument("--learning_rate", type=float, help="Learning rate for the model", default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, help="Warmup ratio for the model", default=0.1)
    parser.add_argument("--weight_decay", type=float, help="Weight decay for the model", default=0.01)
    parser.add_argument("--device", type=str, help="Device to train the model", default="cuda")
    parser.add_argument("--evaluation", type=str, help="Evaluation mode for the model: True or False", default="False")
    parser.add_argument("--model_path", type=str, help="Path to the model to evaluate", default="model")
    parser.add_argument("--log_mode", type=str, help="'offline' or 'online' (wandb)", default="offline")
    parser.add_argument("--freeze_base", type=str, help="Whether to freeze the base model", default="False")
    args = parser.parse_args()


    os.environ["WANDB_PROJECT"] = "gen_detector"  # name your W&B project
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

    # set wandb to offline mode
    if args.log_mode == "offline":
        print("Nooo")
        os.environ["WANDB_MODE"] = "offline"
    elif args.log_mode == "online":
        print("whoaa")
        os.environ["WANDB_MODE"] = "online"
    else:
        raise ValueError("Log mode must be either 'offline' or 'online'")
    
    dataset = load_from_disk(args.dataset_path)

    if args.evaluation == "False":
        if args.detector == "roberta":
            #detector_path = "openai-community/roberta-base-openai-detector"
            #detector_path = "FacebookAI/roberta-large"
            detector_path = "FacebookAI/roberta-base"
            detector_model = RobertaForSequenceClassification.from_pretrained(detector_path).to(args.device)
            bert_tokenizer = RobertaTokenizer.from_pretrained(detector_path)
            detector = LLMDetector(detector_model, bert_tokenizer, 2)
        else:
            raise ValueError("No other detector currently supported")

        dataset = dataset.map(lambda x: tokenize_text(x, bert_tokenizer), batched=True)
        run(args.num_epochs, detector_model, bert_tokenizer, dataset, args.learning_rate, args.warmup_ratio, args.weight_decay, args.batch_size)

 
    elif args.evaluation == "True":
        model = RobertaForSequenceClassification.from_pretrained(args.model_path).to(args.device)
        # tokenize text
        detector_path = "openai-community/roberta-base-openai-detector"
        bert_tokenizer = RobertaTokenizer.from_pretrained(detector_path)

        dataset = dataset.map(lambda x: tokenize_text(x, bert_tokenizer), batched=True)
        metric = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        # load trainer
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                per_device_eval_batch_size=8,
                report_to="wandb",
                output_dir="./results",
            ),
            compute_metrics=compute_metrics,
            eval_dataset=dataset["valid"]
        )
        trainer.evaluate()

    else:
        raise ValueError("Evaluation mode must be either True or False")

    


    