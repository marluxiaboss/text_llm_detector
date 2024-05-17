from datasets import concatenate_datasets, load_from_disk, DatasetDict, Dataset, load_dataset
import argparse
import os
import json
import pandas as pd

from zero_shot_detector.test_fast_detect_gpt import predict_on_dataset

if __name__ == "__main__":
    
    # e.g. python src/format_out_of_domain_dataset.py --save_path=fake_true_datasets/xsum_true_only_test --original_dataset_path=EdinburghNLP/xsum --dataset_name=xsum_only_true --take_samples=10000 --orig_dataset_type=huggingface
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, help="Path to save the dataset", required=True)
    parser.add_argument("--original_dataset_path", type=str, help="Path to the original json dataset", required=True)
    parser.add_argument("--orig_dataset_type", type=str, help="Type of the original dataset (between huggingface and json)", required=True)
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset", required=True)
    parser.add_argument("--take_samples", type=int, help="Number of samples to take from the dataset", required=False, default=None)
    args = parser.parse_args()

    
    # plan:
    # 1) Load fake news articles dataset (10k samples)
    # 2) Run FastDetectGPT to create labels for the datasetl. 0 = true -> accepted, 1 = fake -> rejected
    # 3) Format into DPO format
    
    # 1) Load fake news articles dataset (10k samples)
    dataset_path = args.dataset_path

    # load the dataset, we only take the test split
    dataset_full = load_from_disk(dataset_path)

    if args.test_only:
        dataset = dataset_full["test"]

        if args.take_samples > 0:
            dataset = dataset.select(range(args.take_samples))
    else:
        dataset = dataset_full
        
        
    # 2) Run FastDetectGPT to create labels for the datasetl. 0 = true -> accepted, 1 = fake -> rejected
    preds, labels = predict_on_dataset(dataset)
    
    print("preds", preds)
    print("labels", labels)
        
    """
    # 3) Format into DPO format    
    
    # print first rows of the dataset to verify the validity
    print(fake_true_dataset_df.head())
    
    # transform back the pandas dataframe into a huggingface dataset and save it
    fake_true_dataset = DatasetDict({"test": Dataset.from_pandas(fake_true_dataset_df)})
    fake_true_dataset.save_to_disk(f"{args.save_path}")

    # save the json version of the dataset
    fake_true_dataset_df.to_json(f"{args.save_path}.json", force_ascii=False, indent=4)
    

    training_args = DPOConfig(
    beta=0.1,
    )
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    """