from datasets import concatenate_datasets, load_from_disk, DatasetDict
import argparse
import os
import pandas as pd

from abc import ABC, abstractmethod


class Parphraser(ABC):

    @abstractmethod
    def paraphrase(self, text: str) -> str:
        pass

    @abstractmethod
    def batch_paraphrase(self, texts: list) -> list:
        pass


class LLMParaphraser(Parphraser):

    def __init__(self, tokenizer, model, device, n_paraphrasing=1):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.n_paraphrasing = n_paraphrasing

class CharacterFilter(ABC):
    
    def __init__(self):
        pass

    def filter(self, text: str) -> str:
        pass

class SingleCharacterFilter(CharacterFilter):
    
    def __init__(self, char, replacement_char=""):
        self.char = char
        self.replacement_char = replacement_char

    def filter(self, text: str) -> str:
        return text.replace(self.char, self.replacement_char)
    

"""
TODO: figure out how to correctly remove what's in between parenthesis without shortening the text
class InBetweenCharacterFilter(CharacterFilter):
    
    def __init__(self, char, replacement_char=""):
        self.char = char
        self.replacement_char = replacement_char

    def filter(self, text: str) -> str:
        return text.replace(self.char, f"{self.char}{self.replacement_char}{self.char}")
"""


def apply_character_filters(text: str, character_filters: list) -> str:
    for character_filter in character_filters:
        text = character_filter.filter(text)
    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, help="Path to the fake true dataset (generated with generate_fake_true_dataset.py)", default="fake_true_dataset")
    parser.add_argument("--normalize_apostrophes", help="Whether to normalize apostrphes", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--normalize_quotes", help="Whether to normalize quotes", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--dataset_name_suffix", type=str, help="Suffix to add to the dataset name", default="new")
    parser.add_argument("--test_only", help="Whether to only keep the test split", default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    dataset_path = args.dataset_path

    # load the dataset, we only take the test split
    dataset_full = load_from_disk(dataset_path)

    if args.test_only:
        dataset = dataset_full["test"]
    else:
        dataset = dataset_full


    # apply transformations

    character_filters = []

    if args.normalize_apostrophes:
        character_filters.append(SingleCharacterFilter("’", "'"))

    if args.normalize_quotes:
        character_filters.append(SingleCharacterFilter("”", "\""))
        character_filters.append(SingleCharacterFilter("“", "\""))

    # apply the character filters
    dataset = dataset.map(lambda x: {"text": apply_character_filters(x["text"], character_filters)})
    
    # save the results to a subfolder of the original dataset
    modified_dataset_folder_base = dataset_path.split("/")[0] + "/modified_datasets"
    dataset_name = dataset_path.split("/")[-1]
    dataset.save_to_disk(f"{modified_dataset_folder_base}/{dataset_name}_{args.dataset_name_suffix}")

    # load to pandas train split
    df_train = pd.DataFrame(dataset['train'])
    df_eval = pd.DataFrame(dataset['valid'])
    df_test = pd.DataFrame(dataset['test'])

    # transform text to list by splitting on \n
    df_train["text"] = df_train["text"].apply(lambda x: x.split("\n"))
    df_eval["text"] = df_eval["text"].apply(lambda x: x.split("\n"))
    df_test["text"] = df_test["text"].apply(lambda x: x.split("\n"))

    # dump to json
    df_train.to_json(f"{modified_dataset_folder_base}/{dataset_name}_{args.dataset_name_suffix}_train.json", force_ascii=False, indent=4)
    df_eval.to_json(f"{modified_dataset_folder_base}/{dataset_name}_{args.dataset_name_suffix}_eval.json", force_ascii=False, indent=4)
    df_test.to_json(f"{modified_dataset_folder_base}/{dataset_name}_{args.dataset_name_suffix}_test.json", force_ascii=False, indent=4)


