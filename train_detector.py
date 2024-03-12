import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np
from transformers import (AutoModelForCausalLM, AutoTokenizer, BertForSequenceClassification, BertTokenizer, BertModel,
 RobertaForSequenceClassification, RobertaTokenizer, RobertaModel, TrainingArguments, Trainer, DataCollatorWithPadding,
    TrainerCallback, ElectraForSequenceClassification, ElectraTokenizer, T5ForSequenceClassification, T5Tokenizer, get_scheduler)
from torch.optim import AdamW
from copy import deepcopy
from tqdm import tqdm
from accelerate import Accelerator
import math

import seaborn as sns
import matplotlib.pyplot as plt


from datasets import load_dataset, load_from_disk, Dataset, DatasetDict, concatenate_datasets
import evaluate
import wandb
import os
import argparse

from datetime import datetime

from detector import LLMDetector



def tokenize_text(x, tokenizer):
    return tokenizer(x["text"], truncation=True, padding="max_length", return_tensors="pt")

def run(num_epochs, model, tokenizer, dataset, learning_rate, warmup_ratio, weight_decay, batch_size, save_dir):

    eval_acc_logs = []
    eval_steps = 0.05
    total_nb_steps = len(dataset["train"]) * num_epochs
    training_steps = int(total_nb_steps // batch_size)

    # list of number of steps at which to evaluate the model
    eval_milestones = list(range(0, training_steps, int(training_steps * eval_steps)))
    eval_milestones_samples = [step * batch_size for step in eval_milestones]

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred

        # special case for the T5 model
        if type(logits) is tuple:
            logits = logits[0]

        predictions = np.argmax(logits, axis=-1)
        print("labels", labels)
        print("predictions", predictions)
        acc = metric.compute(predictions=predictions, references=labels)
        eval_acc_logs.append(acc)
        return acc
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        logging_steps=0.05,
        eval_steps=eval_steps,
        # This is important to set evaluation strategy, otherwise there will be no evaluation
        evaluation_strategy="steps",
        save_steps=0.2,
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

    eval_acc_logs_milestones_samples = {
        eval_milestones_samples[i]: eval_acc_logs[i] for i in range(len(eval_milestones_samples))
    }
    print("eval_acc_logs", eval_acc_logs_milestones_samples)



def process_tokenized_dataset(dataset):
    dataset = dataset.remove_columns(["text"])
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch")
    return dataset


def run_training_loop(num_epochs, model, tokenizer, train_dataset, val_dataset, learning_rate, warmup_ratio, weight_decay, batch_size, save_dir):

    log = create_logger(save_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    #shuffle the datasets
    #train_dataset = train_dataset.shuffle(seed=42)
    #val_dataset = val_dataset.shuffle(seed=42)

    # process both datasets
    train_dataset = process_tokenized_dataset(train_dataset)
    val_dataset = process_tokenized_dataset(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


    optimizer = AdamW((param for param in model.parameters() if param.requires_grad), lr=learning_rate, weight_decay=weight_decay)
    #scheduler = lr_scheduler.LinearLR(optimizer, warmup_ratio)
    
    accelerator = Accelerator(mixed_precision='fp16')

    num_training_steps = math.ceil(num_epochs * len(train_loader))
    warmup_steps = math.ceil(num_training_steps * warmup_ratio)
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)


    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
    val_loader = accelerator.prepare(val_loader)

    # how many samples seen before evaluating the model per epoch
    log_loss_steps = 200
    eval_steps= 1000

    # round both up to the nearest multiple of batch_size
    log_loss_steps = (log_loss_steps + batch_size - 1) // batch_size * batch_size
    eval_steps = (eval_steps + batch_size - 1) // batch_size * batch_size
    print("log_loss_steps", log_loss_steps)
    print("eval_steps", eval_steps)

    eval_acc_logs = []

    max_num_model_save = 3

    # ordered list of best models with list of dicts {nb_samples: int, eval_acc: float, model_path: str}
    # where model_path is "./trained_models/model_{rank_in_list}_{nb_samples}.pt"
    models_ranking = [None for i in range(max_num_model_save)]

    
    num_training_steps = len(train_loader) * num_epochs
    progress_bar = tqdm(range(num_training_steps))
    run = wandb.init()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for i, batch in enumerate(train_loader):

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss
            running_loss += loss.detach().item()

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            progress_bar.update(1)

            if ((i + 1) * batch_size) % log_loss_steps == 0:
                avg_loss = running_loss/ (i + 1)
                print(f'Epoch {epoch+1}/{num_epochs}, Loss after {i*batch_size} samples: {avg_loss:.4f}')
                metrics = {}
                metrics["train/loss"] = avg_loss
                metrics["train/learning_rate"] = scheduler.get_last_lr()[0]
                metrics["train/epoch"] = epoch

                run.log(metrics, step=i)

                running_loss = 0

            if ((i + 1) * batch_size) % eval_steps == 0:
                model.eval()
                total, correct = 0, 0
                for batch in val_loader:
                    with torch.no_grad():
                        input_ids = batch["input_ids"]
                        labels = batch["labels"]
                        attention_mask = batch["attention_mask"]
                        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                        _, predicted = torch.max(outputs.logits, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                eval_acc = correct / total
                print(f'Epoch {epoch+1}/{num_epochs}, Validation accuracy after {i*batch_size} samples: {eval_acc:.4f}')
                run.log({"eval/accuracy": eval_acc}, step=i)
                eval_acc_logs.append({"samples": i*batch_size, "accuracy": eval_acc})

                # save the model if it is in the top 3
                if models_ranking[-1] is None or eval_acc > models_ranking[-1]["eval_acc"]:
                    # find the rank of the model
                    rank = 3
                    for j in range(3):
                        if models_ranking[-1] is None or eval_acc > models_ranking[j]["eval_acc"]:
                            rank = j + 1
                            # move the other models down
                            for k in range(2, j, -1):
                                if models_ranking[k-1] is not None:
                                    models_ranking[k] = models_ranking[k-1]
                                    # rename the model path of the models
                                    model_path = models_ranking[k]["model_path"]
                                    new_model_path = f"./trained_models/model_{k}_{models_ranking[k]['samples']}.pt"
                                    os.rename(model_path, new_model_path)
                                    models_ranking[k]["model_path"] = new_model_path
                            break
                    
                    model_path = f"./trained_models/model_{rank}_{i*batch_size}.pt"
                    torch.save(model.state_dict(), model_path)

    print("eval_acc_logs", eval_acc_logs)
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))
    run.finish()
    plot_nb_samples_metrics(eval_acc_logs)






def plot_nb_samples_metrics(eval_acc_logs):
    # lineplot with nb_samples on x-axis and some metric on y-axis like accuracy
    sns.lineplot(x="samples", y="accuracy", data=eval_acc_logs)

    # save the plot
    plt.savefig("result_plots/accuracy_vs_nb_samples.png")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--detector", type=str, help="Name of the model to train from: roberta, bert, e5", default="roberta")
    parser.add_argument("--dataset_path", type=str, help="Path to the fake true dataset (generated with generate_fake_true_dataset.py)", default="fake_true_dataset")
    parser.add_argument("--batch_size", type=int, help="Batch size to train the model", default=8)
    parser.add_argument("--num_epochs", type=int, help="Number of epochs to train the model", default=3)
    parser.add_argument("--learning_rate", type=float, help="Learning rate for the model", default=1e-3)
    parser.add_argument("--warmup_ratio", type=float, help="Warmup ratio for the model", default=0.1)
    parser.add_argument("--weight_decay", type=float, help="Weight decay for the model", default=0.01)
    parser.add_argument("--device", type=str, help="Device to train the model", default="cuda")
    parser.add_argument("--evaluation", type=str, help="Evaluation mode for the model: True or False", default="False")
    parser.add_argument("--model_path", type=str, help="Path to the model to evaluate", default="model")
    parser.add_argument("--log_mode", type=str, help="'offline' or 'online' (wandb)", default="offline")
    parser.add_argument("--freeze_base", type=str, help="Whether to freeze the base model", default="False")
    parser.add_argument("--save_dir", type=str, help="Directory to save the model", default="./outputs")
    args = parser.parse_args()


    os.environ["WANDB_PROJECT"] = "gen_detector"  # name your W&B project
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

    # set wandb to offline mode
    if args.log_mode == "offline":
        os.environ["WANDB_MODE"] = "offline"
    elif args.log_mode == "online":
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

        elif args.detector == "bert":
            detector_path = "bert-base-uncased"
            detector_model = BertForSequenceClassification.from_pretrained(detector_path).to(args.device)
            bert_tokenizer = BertTokenizer.from_pretrained(detector_path)
            detector = LLMDetector(detector_model, bert_tokenizer, 2)

        elif args.detector == "electra":
            detector_path = "google/electra-base-discriminator"
            detector_model = ElectraForSequenceClassification.from_pretrained(detector_path).to(args.device)
            bert_tokenizer = ElectraTokenizer.from_pretrained(detector_path)
            detector = LLMDetector(detector_model, bert_tokenizer, 2)

        elif args.detector == "t5":
            detector_path = "google-t5/t5-base"
            detector_model = T5ForSequenceClassification.from_pretrained(detector_path).to(args.device)
            bert_tokenizer = T5Tokenizer.from_pretrained(detector_path)
            detector = LLMDetector(detector_model, bert_tokenizer, 2)

        else:
            raise ValueError("No other detector currently supported")
        

        if args.freeze_base == "True":
            LLMDetector.freeze_base(detector_model)

        dataset = dataset.map(lambda x: tokenize_text(x, bert_tokenizer), batched=True)
        #run(args.num_epochs, detector_model, bert_tokenizer, dataset, args.learning_rate, args.warmup_ratio, args.weight_decay, args.batch_size, args.save_dir)
        run_training_loop(args.num_epochs, detector_model, bert_tokenizer, dataset["train"], dataset["valid"], args.learning_rate, args.warmup_ratio, args.weight_decay, args.batch_size, args.save_dir)


 
    elif args.evaluation == "True":
        model = RobertaForSequenceClassification.from_pretrained(args.model_path).to(args.device)
        # tokenize text
        detector_path = "openai-community/roberta-base-openai-detector"
        bert_tokenizer = RobertaTokenizer.from_pretrained(detector_path)

        dataset = dataset.map(lambda x: tokenize_text(x, bert_tokenizer), batched=True)
        metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        # load trainer
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                per_device_eval_batch_size=8,
                #report_to="wandb",
                output_dir="./results",
            ),
            compute_metrics=compute_metrics,
            eval_dataset=dataset["test"]
        )
        
        predictions = trainer.predict(dataset["test"])
        preds = np.argmax(predictions.predictions, axis=-1)
        results = metric.compute(predictions=preds, references=predictions.label_ids)
        
        print("Test metrics:")
        for key, value in results.items():
            print(f"{key}: {value}")
    else:
        raise ValueError("Evaluation mode must be either True or False")

    


    