from datasets import concatenate_datasets, load_from_disk, DatasetDict
import argparse
import os
import pandas as pd

import torch
import nltk.data
nltk.download('punkt')
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from tqdm import tqdm

from abc import ABC, abstractmethod


class Parphraser(ABC):

    @abstractmethod
    def paraphrase(self, text: str) -> str:
        pass

    @abstractmethod
    def batch_paraphrase(self, texts: list, batch_size: int) -> list:
        pass


class LLMParaphraser(Parphraser):

    def __init__(self, tokenizer, model, device, n_paraphrasing=1):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.n_paraphrasing = n_paraphrasing

    def paraphrase(self, text: str) -> str:
        pass

    def batch_paraphrase(self, texts: list, batch_size: int) -> list:
        pass
class HumarinpParaphraser(LLMParaphraser):

    def __init__(self, tokenizer, model, device, n_paraphrasing=1):
        super().__init__(tokenizer, model, device, n_paraphrasing)
        self.nltk_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def paraphrase(
        self,
        text,
        num_beams=5,
        num_beam_groups=5,
        num_return_sequences=1,
        repetition_penalty=10.0,
        diversity_penalty=3.0,
        no_repeat_ngram_size=2,
        temperature=0.7,
        max_length=128
    ):
        sentences = self.nltk_tokenizer.tokenize(text)
        results_text = []
        for i, sentence in enumerate(sentences):

            text = sentence.strip()
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
            results_text.append(res[0])
        res = " ".join(results_text)

        return res
    
    def batch_paraphrase(
        self,
        texts: list,
        batch_size=4,
        num_beams=5,
        num_beam_groups=5,
        num_return_sequences=1,
        repetition_penalty=10.0,
        diversity_penalty=3.0,
        no_repeat_ngram_size=2,
        temperature=0.7,
        max_length=128                  
    ) -> list:
        # same as paraphrase, but with a list of text

        sentences_list = [self.nltk_tokenizer.tokenize(text) for text in texts]
        results_text = []

        #for i in tqdm(range(0, len(sentences_list), batch_size), desc="Paraphrasing..."):
        for i in range(0, len(sentences_list), batch_size):
            
            # if we are at the end of the list, we take the remaining elements
            if i + batch_size > len(sentences_list):
                batch = sentences_list[i:]
            else:
                batch = sentences_list[i:i+batch_size]

            input_ids = self.tokenizer.batch_encode_plus(
                [f'paraphrase: {sentence.strip()}' for sentences in batch for sentence in sentences],
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
            results_text.extend(res)

        # join the results
        results_text = [" ".join(results_text[i:i+len(sentences)]) for i, sentences in enumerate(sentences_list)]

        return results_text

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

class ArticleGenerator:

    """
    Generates news article given a prefix, a model and an optional prompt
    """

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_articles(self, prefixes: list, prompt: str = "", max_length=1024, batch_size=4) -> list:

        articles = []

        for prefix in prefixes:
            input_ids = self.tokenizer(
                f"{prompt} {prefix}",
                return_tensors="pt", padding="longest",
                max_length=max_length,
                truncation=True,
            ).input_ids.to(self.device)

            outputs = self.model.generate(
                input_ids, max_length=max_length
            )

            res = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            articles.extend(res)

        return articles



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
    parser.add_argument("--use_article_generator", help="Whether to use the article generator", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--prompt", type=str, help="Prompt to use for the article generator", default="")
    parser.add_argument("--nb_paraphrasing", type=int, help="Number of paraphrasing to do", default=1)
    parser.add_argument("--take_samples", type=int, help="Number of samples to take from the dataset", default=-1)
    parser.add_argument("--batch_size", type=int, help="Batch size for the paraphrasing", default=4)
    args = parser.parse_args()

    dataset_path = args.dataset_path

    # load the dataset, we only take the test split
    dataset_full = load_from_disk(dataset_path)



    if args.test_only:
        dataset = dataset_full["test"]

        if args.take_samples > 0:
            dataset = dataset.select(range(args.take_samples))
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

    # only keep fake samples
    fake_dataset = dataset.filter(lambda x: x["label"] == 1)

    if args.use_humarin_paraphraser:
        print("Using HumarinP paraphraser")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
        model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base", torch_dtype=torch.bfloat16).to(device)
        paraphraser = HumarinpParaphraser(tokenizer, model, device, n_paraphrasing=args.nb_paraphrasing)

        # apply the paraphraser
        for i in range(paraphraser.n_paraphrasing):

            #dataset = dataset.map(lambda x: {"text": paraphraser.paraphrase(x["text"])}, batched=True, batch_size=16)
            fake_dataset = fake_dataset.map(lambda x: {"text": paraphraser.batch_paraphrase(x["text"], args.batch_size)}, batched=True, batch_size=args.batch_size)
    
    if args.use_article_generator:
        print("Using article generator")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # load model
        model = ...
        tokenizer = ...

        article_generator = ArticleGenerator(model, tokenizer, device)

        # take the prefixes from the dataset
        dataset_list = dataset["text"]
        prefixes = [text[:10] for text in dataset_list]

        if use_chat_template:
            if sample["context"] != "":
                text_instruction = f"Context: {sample['context']} \n {prompt} {sample['instruction']}"
            else:
                text_instruction = f"{prompt} {sample['instruction']}"
            
            match template_type:
                case "system_user":
                    messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"{text_instruction}"},
                    ]
                case "user":
                    messages = [
                    {"role": "user", "content": f"{text_instruction}"},
                    ]
                case _:
                    raise ValueError("Template type not supported")

            text_template = gen_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # force prefix on the generated response
            text_template = f"{text_template}\n{sample["instruction"]}"
            else:
                text_instruction = f"{prompt} {sample["instruction"]}"
                text_template = text_instruction
            return {"text_template": text_template}
        
        true_dataset = true_dataset.map(lambda x: transform_chat_template(x, use_chat_template=use_chat_template))
        true_dataset_list = true_dataset["text_template"]

        if args.prompt != "":
            # add prompt to prefix

        # apply chat template if needed

        # generate articles
        articles = article_generator.generate_articles(dataset_list)




        
    # save the results to a subfolder of the original dataset
    modified_dataset_folder_base = dataset_path.split("/")[0] + "/modified_datasets"
    dataset_name = dataset_path.split("/")[-1]
    dataset.save_to_disk(f"{modified_dataset_folder_base}/{dataset_name}_{args.dataset_name_suffix}")

    if args.test_only:
        df_test = pd.DataFrame(dataset)
        #df_test["text"] = df_test["text"].apply(lambda x: x.split("\n"))
        df_test.to_json(f"{modified_dataset_folder_base}/{dataset_name}_{args.dataset_name_suffix}_test.json", force_ascii=False, indent=4)
    # load to pandas train split
    else:
        df_train = pd.DataFrame(dataset['train'])
        df_eval = pd.DataFrame(dataset['valid'])

        # transform text to list by splitting on \n
        df_train["text"] = df_train["text"].apply(lambda x: x.split("\n"))
        df_eval["text"] = df_eval["text"].apply(lambda x: x.split("\n"))

        # dump to json
        df_train.to_json(f"{modified_dataset_folder_base}/{dataset_name}_{args.dataset_name_suffix}_train.json", force_ascii=False, indent=4)
        df_eval.to_json(f"{modified_dataset_folder_base}/{dataset_name}_{args.dataset_name_suffix}_eval.json", force_ascii=False, indent=4)

        df_test = pd.DataFrame(dataset['test'])
        df_test["text"] = df_test["text"].apply(lambda x: x.split("\n"))
        df_test.to_json(f"{modified_dataset_folder_base}/{dataset_name}_{args.dataset_name_suffix}_test.json", force_ascii=False, indent=4)



    



