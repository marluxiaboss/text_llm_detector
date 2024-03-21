import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np
from transformers import (AutoModelForCausalLM, AutoTokenizer, BertForSequenceClassification, BertTokenizer, BertModel,
 RobertaForSequenceClassification, RobertaTokenizer, RobertaModel, TrainingArguments, Trainer, DataCollatorWithPadding,
    TrainerCallback, ElectraForSequenceClassification, ElectraTokenizer, T5ForSequenceClassification, T5Tokenizer, get_scheduler,
    RobertaConfig, AutoConfig, AutoModelForMaskedLM)
from torch.optim import AdamW
from copy import deepcopy
from tqdm import tqdm
from accelerate import Accelerator
import math

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


from datasets import load_dataset, load_from_disk, Dataset, DatasetDict, concatenate_datasets
import evaluate
import wandb
import os
import argparse
import sys

from datetime import datetime

from detector import LLMDetector
from utils import create_logger, Signal



def tokenize_text(x, tokenizer):
    return tokenizer(x["text"], truncation=True, padding="max_length", return_tensors="pt")


def prepare_dataset_for_checking_degradation(detector_name, tokenizer, batch_size, seed):

    fact_completion_dataset = load_dataset('Polyglot-or-Not/Fact-Completion')["English"]

    # filter out questions where the answer is not a single word for the tokenizer
    len_before = len(fact_completion_dataset)
    #fact_completion_dataset = fact_completion_dataset.filter(lambda x: len(x["true"].split()) == 1)

    # this number seems to be 3 for all models of interest
    single_word_answer_token_size = 3
    fact_completion_dataset = fact_completion_dataset.filter(lambda x: len(tokenizer(x["true"]).input_ids) == single_word_answer_token_size)
    len_after = len(fact_completion_dataset)
    #print(f"Filtered out {len_before - len_after} questions out of {len_before}")

    nb_samples_test_degradation = 1000
    fact_completion_dataset = fact_completion_dataset.shuffle(seed=seed)
    fact_completion_dataset = fact_completion_dataset.select(range(nb_samples_test_degradation))
    questions = fact_completion_dataset["stem"]
    answers = fact_completion_dataset["true"]

    # add [MASK] at the end of each question
    # we also add a ".", this is very important for bert models

    if detector_name == "bert":
        questions_masked = [q + " [MASK]." for q in questions]

    elif detector_name == "roberta":
        questions_masked = [q + " <mask>." for q in questions]

    else:
        raise ValueError("No other detector currently supported")
    

    batches_questions_masked = [questions_masked[i:i + batch_size] for i in range(0, len(questions_masked), batch_size)]
    question_with_answers = [questions[i] + " " + answers[i] + "." for i in range(len(questions))]
    batches_answers = [question_with_answers[i:i + batch_size] for i in range(0, len(question_with_answers), batch_size)]

    batches = list(zip(batches_questions_masked, batches_answers))
    
    return batches

def create_mlm_model(model_name, device):
    if model_name == "roberta":
        model = AutoModelForMaskedLM.from_pretrained("roberta-base").to(device)
    
    if model_name == "bert":
        model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased").to(device)
    
    if model_name == "electra":
        model = AutoModelForMaskedLM.from_pretrained("google/electra-base-discriminator").to(device)

    if model_name == "t5":
        model = AutoModelForMaskedLM.from_pretrained("google-t5/t5-base").to(device)
    
    return model


def adapt_model_to_mlm(classif_model, mlm_model, model_name):

    # transfer the roberta weights to the MaskedLM model
    if model_name == "bert":
        mlm_model.bert = classif_model.bert

    elif model_name == "roberta":
        mlm_model.roberta = classif_model.roberta
    else:
        raise ValueError("No other detector currently supported")
    
    return mlm_model



def check_model_degradation(classif_model, tokenizer, batches, nb_samples, log, mlm_model, model_name):


    model = adapt_model_to_mlm(classif_model, mlm_model, model_name)

    losses = []
    model.eval()

    with torch.no_grad():
        for batch_question, batch_answer in tqdm(batches, desc="Answering fact completion questions..."):
        #for batch_question, batch_answer in batches:
            inputs = tokenizer(batch_question, return_tensors='pt', padding=True, truncation=True).to(model.device)

            questions_with_answer = batch_answer
            #questions_with_answer = [questions[i] + " " + answers[i] + "." for i in range(len(batch))]

            labels = tokenizer(questions_with_answer, return_tensors='pt', padding=True, truncation=True)["input_ids"].to(model.device)
            labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    log.info(f"Average Loss on fact answering task after {nb_samples} samples: {avg_loss:.4f}")

    return avg_loss

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

def create_experiment_folder(model_name, experiment_args):

    # create a folder with the folowing elements:
    # log, description of training with all the args, model folder with all the saved models

    base_path = experiment_args.save_dir
    # check if there exists a subfolder already for the model_name
    if not os.path.isdir(f"{base_path}/{model_name}"):
        os.makedirs(f"{base_path}/{model_name}")
    
    experiment_path = f"{base_path}/{model_name}/{datetime.now().strftime('%d_%m_%H%M')}"
    os.makedirs(experiment_path)
    experiment_saved_model_path = f"{experiment_path}/saved_models"
    os.makedirs(experiment_saved_model_path)
    plots_path = f"{experiment_path}/plots"
    os.makedirs(plots_path)
    test_path = f"{experiment_path}/test"
    os.makedirs(test_path)

    # create a file args_log.txt with all the args
    with open(f"{experiment_path}/args_log.txt", "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"dataset_path: {experiment_args.dataset_path}\n")
        f.write(f"num_epochs: {experiment_args.num_epochs}\n")
        f.write(f"batch_size: {experiment_args.batch_size}\n")
        f.write(f"learning_rate: {experiment_args.learning_rate}\n")
        f.write(f"warmup_ratio: {experiment_args.warmup_ratio}\n")
        f.write(f"weight_decay: {experiment_args.weight_decay}\n")
        f.write(f"device: {experiment_args.device}\n")
        f.write(f"evaluation: {experiment_args.evaluation}\n")
        f.write(f"model_path: {experiment_args.model_path}\n")
        f.write(f"log_mode: {experiment_args.log_mode}\n")
        f.write(f"freeze_base: {experiment_args.freeze_base}\n")
        f.write(f"save_dir: {experiment_args.save_dir}\n")

    return experiment_path


def run_training_loop(num_epochs, model, tokenizer, train_dataset, val_dataset,
                       learning_rate, warmup_ratio, weight_decay, batch_size,
                        save_dir, detector_name, experiment_path, dataset_path,
                        fp16=True, log=None, check_degradation=0, degradation_check_batches=None,
                        mlm_model=None, log_loss_steps=200, eval_steps=500, freeze_base=False):

    experiment_saved_model_path = f"{experiment_path}/saved_models"
    sig = Signal("run_signal.txt")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    #shuffle the datasets
    #train_dataset = train_dataset.shuffle(seed=42)
    #val_dataset = val_dataset.shuffle(seed=42)

    # process both datasets
    train_dataset = process_tokenized_dataset(train_dataset)
    val_dataset = process_tokenized_dataset(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


    optimizer = AdamW((param for param in model.parameters() if param.requires_grad), lr=learning_rate, weight_decay=weight_decay)
    #scheduler = lr_scheduler.LinearLR(optimizer, warmup_ratio)
    
    if fp16:
        accelerator = Accelerator(mixed_precision='fp16')
    else:
        accelerator = Accelerator()
    

    num_training_steps = math.ceil(num_epochs * len(train_loader))
    warmup_steps = math.ceil(num_training_steps * warmup_ratio)
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)


    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
    val_loader = accelerator.prepare(val_loader)

    # round both up to the nearest multiple of batch_size
    log_loss_steps = (log_loss_steps + batch_size - 1) // batch_size * batch_size
    eval_steps = (eval_steps + batch_size - 1) // batch_size * batch_size
    log.info(f"log_loss_steps: {log_loss_steps}")
    log.info(f"eval_steps: {eval_steps}")

    eval_acc_logs = []
    train_loss_logs = []
    loss_degradation_logs = []

    best_model = None

    
    num_training_steps = len(train_loader) * num_epochs
    progress_bar = tqdm(range(num_training_steps))

    tags = [detector_name, dataset_path]
    if fp16:
        tags.append("fp16")

    if freeze_base == "True":
        tags.append("freeze_base")
    else:
        tags.append("full_finetuning")

    run = wandb.init(project="detector_training", tags=tags, dir=experiment_path)


    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        log.info(f"----------------Epoch {epoch+1}/{num_epochs}----------------")
        sig.update()
        if sig.training_sig:
            for i, batch in enumerate(train_loader):
                sig.update()

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
                    nb_steps_log = log_loss_steps // batch_size
                    avg_loss = running_loss / nb_steps_log
                    nb_samples_seen = i*batch_size + epoch*len(train_loader)*batch_size
                    log.info(f'Epoch {epoch+1}/{num_epochs}, Loss after {nb_samples_seen} samples: {avg_loss:.4f}')
                    train_loss_logs.append({"samples": nb_samples_seen, "loss": avg_loss})
                    metrics = {}
                    metrics["train/loss"] = avg_loss
                    metrics["train/learning_rate"] = scheduler.get_last_lr()[0]
                    metrics["train/epoch"] = epoch

                    run.log(metrics, step=nb_samples_seen)

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
                    nb_samples_seen = i*batch_size + epoch*len(train_loader)*batch_size
                    log.info(f'Epoch {epoch+1}/{num_epochs}, Validation accuracy after {nb_samples_seen} samples: {eval_acc:.4f}')
                    run.log({"eval/accuracy": eval_acc}, step=nb_samples_seen)
                    eval_acc_logs.append({"samples": nb_samples_seen, "accuracy": eval_acc})

                    if best_model is None or eval_acc > best_model["eval_acc"]:
                        best_model = {"eval_acc": eval_acc, "nb_samples": nb_samples_seen}
                        
                        # save the model
                        torch.save(model.state_dict(), f"{experiment_saved_model_path}/best_model.pt")
                        log.info(f"Best model with eval accuracy {eval_acc} with {nb_samples_seen} samples seen is saved")
                    model.train()
                               
                if check_degradation > 0 and ((i + 1) * batch_size) % check_degradation == 0:
                    nb_samples_seen = i*batch_size + epoch*len(train_loader)*batch_size
                    degradation_loss = check_model_degradation(model, tokenizer, degradation_check_batches, nb_samples_seen, log, mlm_model, detector_name)
                    loss_degradation_logs.append({"samples": nb_samples_seen, "loss": degradation_loss})


        else:
            log.info("Training signal is False, stopping training")
            break
    
    log.info(f"Training finished")
    log.info(f"Best model: {best_model}")
    log.info(f"Training loss logs: {train_loss_logs}")
    log.info(f"Evaluation accuracy logs: {eval_acc_logs}")
    run.finish()
    plot_nb_samples_metrics(eval_acc_logs, save_path=f"{experiment_path}/plots")
    plot_nb_samples_loss(train_loss_logs, save_path=f"{experiment_path}/plots")

    if check_degradation > 0:
        plot_degradation_loss(loss_degradation_logs, save_path=f"{experiment_path}/plots")


def plot_nb_samples_metrics(eval_acc_logs, save_path):
    # transform to df
    eval_acc_logs_df = pd.DataFrame(eval_acc_logs)
    plt.figure()
    # lineplot with nb_samples on x-axis and some metric on y-axis like accuracy
    sns.lineplot(x="samples", y="accuracy", data=eval_acc_logs_df)

    # save the plot
    plt.savefig(f"{save_path}/accuracy_vs_nb_samples.png")

def plot_nb_samples_loss(train_loss_logs, save_path):
    # transform to df
    train_loss_logs_df = pd.DataFrame(train_loss_logs)
    plt.figure()
    # lineplot with nb_samples on x-axis and loss on y-axis
    sns.lineplot(x="samples", y="loss", data=train_loss_logs_df)

    # save the plot
    plt.savefig(f"{save_path}/loss_vs_nb_samples.png")

def plot_degradation_loss(loss_degradation_logs, save_path):
    # transform to df
    loss_degradation_logs_df = pd.DataFrame(loss_degradation_logs)
    plt.figure()
    # lineplot with nb_samples on x-axis and loss on y-axis
    sns.lineplot(x="samples", y="loss", data=loss_degradation_logs_df)

    # save the plot
    plt.savefig(f"{save_path}/degradation_loss_vs_nb_samples.png")


def test_model(model, batch_size, dataset, experiment_path, log, dataset_path):

    # load best model
    #model.load_state_dict(torch.load(f"{experiment_path}/saved_models/best_model.pt"))
    model.eval()
    metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # load trainer
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            per_device_eval_batch_size=batch_size,
            output_dir=f"{experiment_path}/test",
        ),
        compute_metrics=compute_metrics,
        eval_dataset=dataset["test"]
    )
    
    log.info(f"Evaluating the best model on the test set of dataset {dataset_path}...")
    predictions = trainer.predict(dataset["test"])
    preds = np.argmax(predictions.predictions, axis=-1)
    results = metric.compute(predictions=preds, references=predictions.label_ids)
    
    log.info("Test metrics:")
    for key, value in results.items():
        log.info(f"{key}: {value}")


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
    parser.add_argument("--save_dir", type=str, help="Directory to save the model and logs", default="./training_logs/detector_freeze_base")
    parser.add_argument("--fp16", type=str, help="Whether to use fp16", default="True")
    parser.add_argument("--check_degradation", type=int, help="If set to > 0, then check for the degration of the model after each check_degradation steps", default=0)
    parser.add_argument("--log_loss_steps", type=int, help="How many samples seen before logging the loss", default=200)
    parser.add_argument("--eval_steps", type=int, help="How many samples seen before evaluating the model", default=500)
    parser.add_argument("--add_more_layers", type=str, help="Whether to add more layers to the classifier", default="False")
    args = parser.parse_args()


    os.environ["WANDB_PROJECT"] = "detector_training"  # name your W&B project
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
            #detector_path = "google-t5/t5-base"
            # the path above has issues when fp16 is set to True
            detector_path = "google-t5/t5-base"
            detector_model = T5ForSequenceClassification.from_pretrained(detector_path).to(args.device)
            bert_tokenizer = T5Tokenizer.from_pretrained(detector_path)
            detector = LLMDetector(detector_model, bert_tokenizer, 2)

            # set fp16 to False for T5 model, t5 has issues with fp16
            args.fp16 = False

        else:
            raise ValueError("No other detector currently supported")
        

        if args.freeze_base == "True":
            LLMDetector.freeze_base(detector_model)

        if args.add_more_layers == "True":
            LLMDetector.add_more_layers(detector_model)

        dataset = dataset.map(lambda x: tokenize_text(x, bert_tokenizer), batched=True)
        #run(args.num_epochs, detector_model, bert_tokenizer, dataset, args.learning_rate, args.warmup_ratio, args.weight_decay, args.batch_size, args.save_dir)       
        experiment_path = create_experiment_folder(args.detector, args)

        # create log file
        with open(f"{experiment_path}/log.txt", "w") as f:
            f.write("")

        degradation_check_batches = None   
        mlm_model = None
        log = create_logger(__name__, silent=False, to_disk=True,
                                    log_file=f"{experiment_path}/log.txt")
        
        # test model loss on mlm eval before training
        if args.check_degradation > 0:
            degradation_check_batches = prepare_dataset_for_checking_degradation(args.detector, bert_tokenizer, args.batch_size, seed=42)
            mlm_model = create_mlm_model(args.detector, args.device)
            ref_degredation_loss = check_model_degradation(detector_model, bert_tokenizer, degradation_check_batches, args.check_degradation, log, mlm_model, args.detector)
            

        # run the training loop
        run_training_loop(args.num_epochs, detector_model, bert_tokenizer, dataset["train"], dataset["valid"],
                           args.learning_rate, args.warmup_ratio, args.weight_decay, args.batch_size, args.save_dir, args.detector, experiment_path, args.dataset_path,
                           args.fp16, log, args.check_degradation, degradation_check_batches, mlm_model, args.log_loss_steps, args.eval_steps, args.freeze_base)
        
        # evaluate model on the test set after training by loading the best model
        test_model(detector_model, args.batch_size, dataset, experiment_path, log, args.dataset_path)


 
    elif args.evaluation == "True":

        if args.detector == "roberta":
            detector_path = "FacebookAI/roberta-base"
            config = AutoConfig.from_pretrained(detector_path)
            model = RobertaForSequenceClassification(config)
            bert_tokenizer = RobertaTokenizer.from_pretrained(detector_path)

        elif args.detector == "bert":
            detector_path = "bert-base-uncased"
            config = AutoConfig.from_pretrained(detector_path)
            model = BertForSequenceClassification(config)
            bert_tokenizer = BertTokenizer.from_pretrained(detector_path)

        elif args.detector == "electra":
            detector_path = "google/electra-base-discriminator"
            config = AutoConfig.from_pretrained(detector_path)
            detector_model = ElectraForSequenceClassification(config)
            bert_tokenizer = ElectraTokenizer.from_pretrained(detector_path)

        elif args.detector == "t5":
            detector_path = "google-t5/t5-base"
            config = AutoConfig.from_pretrained(detector_path)
            detector_model = T5ForSequenceClassification(config)
            bert_tokenizer = T5Tokenizer.from_pretrained(detector_path)

        else:
            raise ValueError("No other detector currently supported")


        # load the trained weights
        model.load_state_dict(torch.load(args.model_path))
        model = model.to(args.device)

        dataset = dataset.map(lambda x: tokenize_text(x, bert_tokenizer), batched=True)

        # if args.model_path is experiment_path/saved_models/best_model.pt, then the experiment_path is experiment_path
        experiment_path = args.model_path.split("/saved_models")[0]


        # create log file
        with open(f"{experiment_path}/test/log_{args.dataset_path}.txt", "w") as f:
            f.write("")

        log = create_logger(__name__, silent=False, to_disk=True,
                                    log_file=f"{experiment_path}/test/log_{args.dataset_path}.txt")

        test_model(model, args.batch_size, dataset, experiment_path, log, args.dataset_path)
    else:
        raise ValueError("Evaluation mode must be either True or False")

    


    