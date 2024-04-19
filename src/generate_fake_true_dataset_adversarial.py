from datasets import concatenate_datasets, load_from_disk, DatasetDict
import argparse
import os
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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

    def paraphrase(self, text: str) -> str:
        pass

    def batch_paraphrase(self, texts: list) -> list:
        pass
class HumarinpParaphraser(LLMParaphraser):

    def __init__(self, tokenizer, model, device, n_paraphrasing=1):
        super().__init__(tokenizer, model, device, n_paraphrasing)

    def paraphrase(
        self,
        text,
        num_beams=5,
        num_beam_groups=5,
        num_return_sequences=5,
        repetition_penalty=10.0,
        diversity_penalty=3.0,
        no_repeat_ngram_size=2,
        temperature=0.7,
        max_length=128
    ):
        input_ids = self.tokenizer(
            f'paraphrase: {text}',
            return_tensors="pt", padding="longest",
            max_length=max_length,
            truncation=True,
        ).input_ids.to(self.device)
        
        outputs = self.model.generate(
            input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
            num_beams=num_beams, num_beam_groups=num_beam_groups,
            max_length=max_length, diversity_penalty=diversity_penalty
        )

        res = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return res

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
    parser.add_argument("--use_humarin_paraphraser", help="Whether to use the HumarinP model for paraphrasing", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--nb_paraphrasing", type=int, help="Number of paraphrasing to do", default=1)
    parser.add_argument("--take_samples", type=int, help="Number of samples to take from the dataset", default=-1)
    args = parser.parse_args()

    dataset_path = args.dataset_path

    # load the dataset, we only take the test split
    dataset_full = load_from_disk(dataset_path)

    if args.take_samples > 0:
        dataset_full = dataset_full.select(range(args.take_samples))

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

    if character_filters:

        # apply the character filters
        dataset = dataset.map(lambda x: {"text": apply_character_filters(x["text"], character_filters)})


    if args.use_humarin_paraphraser:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
        model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)
        paraphraser = HumarinpParaphraser(tokenizer, model, device, n_paraphrasing=args.nb_paraphrasing)

        # apply the paraphraser
        for i in range(len(paraphraser.n_paraphrasing)):
            dataset = dataset.map(lambda x: {"text": paraphraser.paraphrase(x["text"])})
        
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

    



