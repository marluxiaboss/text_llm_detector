import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
    Trainer,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling
)

import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb_training_samples", type=int, help="Number of training samples", required=True)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=1)
    parser.add_argument("--learning_rate", type=float, help="Learning rate", default=2e-4)
    args = parser.parse_args()

    # Model
    base_model = "microsoft/phi-2"
    #base_model = "openai-community/gpt2"

    new_model = "phi-2-cnn_news"

    # Dataset
    dataset_path = "cnn_dailymail"
    raw_dataset = load_dataset(dataset_path, "3.0.0")["train"]

    # sample 1000 examples 
    raw_dataset = raw_dataset.shuffle(seed=42)
    raw_dataset = raw_dataset.select(range(args.nb_training_samples))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"

    # Tokenize
    def tokenize_function(example):
        return tokenizer(example["article"], padding="max_length", truncation=True)


    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)
    print(tokenized_dataset)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)

    training_arguments = TrainingArguments(
        output_dir = "./phi_finetuning_news_results",
        num_train_epochs = 1,
        fp16 = False,
        bf16 = True,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size = args.batch_size,
        learning_rate = args.learning_rate,
        lr_scheduler_type = "cosine",
        warmup_ratio = 0.1,
        #max_grad_norm = 0.3,
        save_steps = 0,
        logging_steps = 5,
    )


    # Set supervised fine-tuning parameters
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        args=training_arguments,
        data_collator=data_collator,
    )

    # Train model
    trainer.train()

    # Save model
    model.save_pretrained(f"trained_models/{new_model}")