from datasets import concatenate_datasets, load_from_disk, DatasetDict, Dataset
import argparse
import os
import json
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, help="Path to save the dataset", required=True)
    parser.add_argument("--original_json_dataset_path", type=str, help="Path to the original json dataset", required=True)
    args = parser.parse_args()


    # load original json dataset, make it into a dataframe, transform into correct shape, and save it
    orig_json_dataset_path = args.original_json_dataset_path
    dataset_df = pd.read_json(orig_json_dataset_path)

    fake_true_samples = []
    for index, row in dataset_df.iterrows():
        fake_true_samples.append({"text": row["original"], "label": 0})
        fake_true_samples.append({"text": row["sampled"], "label": 1})

    
    #fake_true_dataset = DatasetDict({"test": fake_true_dataset_df})
    fake_true_dataset_df = pd.DataFrame(fake_true_samples)
    fake_true_dataset_df["text"] = fake_true_dataset_df["text"].apply(lambda x: x.split("\n"))
    print(fake_true_dataset_df.head())
    #fake_true_dataset = Dataset.from_pandas(fake_true_dataset_df)
    fake_true_dataset = DatasetDict({"test": Dataset.from_pandas(fake_true_dataset_df)})
    fake_true_dataset.save_to_disk(f"{args.save_path}")

    # save the json version of the dataset
    fake_true_dataset_df.to_json(f"{args.save_path}.json", force_ascii=False, indent=4)