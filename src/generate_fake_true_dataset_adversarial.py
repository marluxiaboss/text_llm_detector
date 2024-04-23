from datasets import concatenate_datasets, load_from_disk, DatasetDict
import argparse
import os
import pandas as pd
import copy

import torch
import nltk.data
nltk.download('punkt')
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_from_disk, concatenate_datasets, Dataset

from abc import ABC, abstractmethod

from model_loader import load_generator


class Parphraser(ABC):
    """
    Abstract class for a paraphraser
    """

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

        results_joined = []
        i = 0
        for sentence in sentences_list:
            len_sent = len(sentence)
            results_joined.append(" ".join(results_text[i:i+len_sent]))
            i += len_sent


        return results_joined

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
    

class ArticleGenerator:

    """
    Generates news article given a prefix, a model and an optional prompt
    """

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_articles(self, prefixes_with_prompt: list, prefixes: list, batch_size=4) -> list:

        articles = []

        for i in range(0, len(prefixes_with_prompt), batch_size):
            if i + batch_size > len(prefixes_with_prompt):
                samples = prefixes_with_prompt[i:]
            else:
                samples = prefixes_with_prompt[i:i+batch_size]
            outputs = self.model(samples)
            res = [text.replace("\n", "") for text in outputs]
            res = [f"{prefixes[j + i]}{res[j]}" for j in range(len(res))]

            articles.extend(res)

        return articles
    
def transform_chat_template_with_prompt(prefix, prompt, tokenizer, use_chat_template=False, template_type=None):

    if prefix != "":
        text_instruction = f"{prompt} {prefix}"
    else:
        text_instruction = prompt
        
    if use_chat_template:
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

        text_template = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # force prefix on the generated response
        text_template = f"{text_template}\n{prefix}"

    else:
        text_template = text_instruction

    return text_template


def regroup_pairs(merged_dataset, seed=42):
    """
    Regroup pairs of true and fake responses two by two so that they are in the same batch and in the same split
    """

    def fix_ids(dataset):
        """
        Fix the ids of the dataset
        """
        fake_responses_dataset = dataset.filter(lambda x: x["label"] == 1)
        true_responses_dataset = dataset.filter(lambda x: x["label"] == 0)

        fake_responses_text = fake_responses_dataset["text"]
        true_responses_text = true_responses_dataset["text"]

        correct_text_ordering_fake = []
        correct_text_ordering_true = []

        for i, _ in enumerate(fake_responses_text):

            fake_response = fake_responses_text[i]

            # find the prefix in true_dataset
            prefix = " ".join(fake_response.split()[:10])

            for j, _ in enumerate(true_responses_text):
                if " ".join(true_responses_text[j].split()[:10]) == prefix:
                    #correct_ids_fake_dataset.append(true_reponses_labels[i])
                    correct_text_ordering_true.append(j)
                    correct_text_ordering_fake.append(i)
                    break   

        # reorganize the fake responses according to the correct order
        fake_responses_dataset = fake_responses_dataset.select(correct_text_ordering_fake)

        # remove true_responses without a corresponding fake response
        true_responses_dataset = true_responses_dataset.select(correct_text_ordering_true)

        # add an id column to fake and true responses datasets
        fake_responses_dataset = fake_responses_dataset.add_column("id", list(range(len(fake_responses_dataset))))
        true_responses_dataset = true_responses_dataset.add_column("id", list(range(len(true_responses_dataset))))
                                                                   
        dataset = concatenate_datasets([true_responses_dataset, fake_responses_dataset])

        # shuffle the dataset to mix between true and fake responses within pairs and sort by id to have the correct order again
        dataset = dataset.shuffle(seed=seed)
        dataset = dataset.sort("id")
        dataset = dataset.remove_columns("id")

        return dataset
    
    # shuffle the dataset
    merged_dataset = merged_dataset.shuffle(seed=seed)

    # ids may be incorrect for label 1, we need to fix them
    merged_dataset = fix_ids(merged_dataset)

    return merged_dataset



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
    parser.add_argument("--article_generator", type=str, help="Generator used to generate the articles, it should be a chat model", default="zephyr")
    parser.add_argument("--prompt", type=str, help="Prompt to use for the article generator", default="")
    parser.add_argument("--nb_paraphrasing", type=int, help="Number of paraphrasing to do", default=1)
    parser.add_argument("--take_samples", type=int, help="Number of samples to take from the dataset", default=-1)
    parser.add_argument("--batch_size", type=int, help="Batch size for the paraphrasing", default=4)
    parser.add_argument("--temperature", type=float, help="Temperature for the generation, default one if not set", default=-1.0)
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

        # copy the original dataset
        fake_dataset_orig = copy.deepcopy(fake_dataset)

        # apply the paraphraser
        for i in range(paraphraser.n_paraphrasing):

            #dataset = dataset.map(lambda x: {"text": paraphraser.paraphrase(x["text"])}, batched=True, batch_size=16)
            fake_dataset = fake_dataset.map(lambda x: {"text": paraphraser.batch_paraphrase(x["text"], args.batch_size)}, batched=True, batch_size=args.batch_size)

        fake_dataset = Dataset.from_dict(({"text": fake_dataset_orig["text"], "fake_paraphrased":  [text[:500] for text in fake_dataset["text"]], "label": [1]*len(fake_dataset)}))
        true_dataset = Dataset.from_dict({"text": dataset.filter(lambda x: x["label"] == 0)["text"], "fake_paraphrased": dataset.filter(lambda x: x["label"] == 0)["text"], "label": [0]*len(dataset.filter(lambda x: x["label"] == 0))})

        dataset = concatenate_datasets([true_dataset, fake_dataset])
        dataset = regroup_pairs(dataset)
        dataset = dataset.map(lambda x: {"text": x["fake_paraphrased"], "label": x["label"]})
        dataset = dataset.remove_columns(["fake_paraphrased"])

    
    if args.use_article_generator:
        generator = args.article_generator
        print(f"Using article generator with {generator}")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, tokenizer, use_chat_template, template_type = load_generator(generator, device, temperature=args.temperature)
        article_generator = ArticleGenerator(model, tokenizer, device)

        # take the prefixes from the dataset
        dataset_list = fake_dataset["text"]
        prefixes = [" ".join(text.split()[:10]) for text in dataset_list]

        # apply the chat template with the prompt
        prefixes_with_prompt = [transform_chat_template_with_prompt(prefix, args.prompt, tokenizer, use_chat_template, template_type) for prefix in prefixes]

        # generate articles
        fake_articles = article_generator.generate_articles(prefixes_with_prompt, prefixes, batch_size=args.batch_size)

        # combine with true article to re-create the fake_true_dataset
        #fake_dataset = fake_dataset.map(lambda x: {"text": fake_articles[x["id"]], "label": 1})
        #fake_dataset_orig = copy.deepcopy(fake_dataset)

        true_dataset = dataset.filter(lambda x: x["label"] == 0)
        fake_dataset = Dataset.from_dict({"text": [text[:500] for text in fake_articles], "label": [1] * len(fake_articles)})
        true_dataset = Dataset.from_dict({"text": true_dataset["text"], "label": [0] * len(fake_articles)})

        # regroup the pairs to re-create the dataset as it was before
        dataset = concatenate_datasets([true_dataset, fake_dataset])
        dataset = regroup_pairs(dataset)

        
    # save the results to a subfolder of the original dataset
    modified_dataset_folder_base = dataset_path.split("/")[0] + "/modified_datasets"
    dataset_name = dataset_path.split("/")[-1]
    dataset.save_to_disk(f"{modified_dataset_folder_base}/{dataset_name}_{args.dataset_name_suffix}")

    if args.test_only:
        df_test = pd.DataFrame(dataset)
        df_test["text"] = df_test["text"].apply(lambda x: x.split("\n"))
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



    



