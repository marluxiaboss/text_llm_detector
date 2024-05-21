from datasets import concatenate_datasets, load_from_disk, DatasetDict, Dataset, load_dataset
import argparse
import os
import json
import pandas as pd
import numpy as np
import torch


from transformers import AutoTokenizer

from trl import (
    DPOConfig,
    DPOTrainer,
)

import sys
SRC_PATH = ["src"]
for module_path in SRC_PATH:
    if module_path not in sys.path:
        sys.path.append(module_path)
        
from utils import *
from model_loader import load_generator



if __name__ == "__main__":
    
    # e.g. python src/format_out_of_domain_dataset.py --save_path=fake_true_datasets/xsum_true_only_test --original_dataset_path=EdinburghNLP/xsum --dataset_name=xsum_only_true --take_samples=10000 --orig_dataset_type=huggingface
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="Path to the original dataset", required=True)
    parser.add_argument("--model_name", type=str, help="Name of the model that will be used for DPO, used for the tokenizer to have the chat template", required=True, default="zephyr")
    args = parser.parse_args()

    
    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = args.model_name
    generator="gpt2"
    model, tokenizer, use_chat_template, template_type = load_generator(generator, device)
    model = model.gpt.to(device)
    
    model_ref = None
    
    # load dataset
    dataset = load_from_disk(args.dataset_path)
    train_dataset = dataset["train"]
    print("train_dataset: ", train_dataset)


    training_args = DPOConfig(
        beta=0.1,
        output_dir="src/dpo_training/dpo_training_output",
    )
    
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
