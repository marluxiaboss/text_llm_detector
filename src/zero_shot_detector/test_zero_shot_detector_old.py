# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random

import numpy as np
import torch
import os
import time
import glob
import argparse
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from torch.utils.data import DataLoader

from tqdm import tqdm



### MODEL FILE ###
def from_pretrained(cls, model_name, kwargs, cache_dir):
    # use local model if it exists
    local_path = os.path.join(cache_dir, 'local.' + model_name.replace("/", "_"))
    if os.path.exists(local_path):
        return cls.from_pretrained(local_path, **kwargs)
    return cls.from_pretrained(model_name, **kwargs, cache_dir=cache_dir)

# predefined models
model_fullnames = {  'gpt2': 'gpt2',
                     'gpt2-xl': 'gpt2-xl',
                     'opt-2.7b': 'facebook/opt-2.7b',
                     'gpt-neo-2.7B': 'EleutherAI/gpt-neo-2.7B',
                     'gpt-j-6B': 'EleutherAI/gpt-j-6B',
                     'gpt-neox-20b': 'EleutherAI/gpt-neox-20b',
                     'mgpt': 'sberbank-ai/mGPT',
                     'pubmedgpt': 'stanford-crfm/pubmedgpt',
                     'mt5-xl': 'google/mt5-xl',
                     'llama-13b': 'huggyllama/llama-13b',
                     'llama2-13b': 'TheBloke/Llama-2-13B-fp16',
                     'bloom-7b1': 'bigscience/bloom-7b1',
                     'opt-13b': 'facebook/opt-13b',
                     }
float16_models = ['gpt-j-6B', 'gpt-neox-20b', 'llama-13b', 'llama2-13b', 'bloom-7b1', 'opt-13b']

def get_model_fullname(model_name):
    return model_fullnames[model_name] if model_name in model_fullnames else model_name

def load_model(model_name, device, cache_dir):
    model_fullname = get_model_fullname(model_name)
    print(f'Loading model {model_fullname}...')
    model_kwargs = {}
    if model_name in float16_models:
        model_kwargs.update(dict(torch_dtype=torch.float16))
    if 'gpt-j' in model_name:
        model_kwargs.update(dict(revision='float16'))
    model = from_pretrained(AutoModelForCausalLM, model_fullname, model_kwargs, cache_dir)
    print('Moving model to GPU...', end='', flush=True)
    start = time.time()
    model.to(device)
    print(f'DONE ({time.time() - start:.2f}s)')
    return model

def load_tokenizer(model_name, for_dataset, cache_dir):
    model_fullname = get_model_fullname(model_name)
    optional_tok_kwargs = {}
    if "facebook/opt-" in model_fullname:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    if for_dataset in ['pubmed']:
        optional_tok_kwargs['padding_side'] = 'left'
    else:
        optional_tok_kwargs['padding_side'] = 'right'
    base_tokenizer = from_pretrained(AutoTokenizer, model_fullname, optional_tok_kwargs, cache_dir=cache_dir)
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
        if '13b' in model_fullname:
            base_tokenizer.pad_token_id = 0
    return base_tokenizer


### FAST DETECT GPT FILE ###
def get_samples(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    nsamples = 10000
    lprobs = torch.log_softmax(logits, dim=-1)
    distrib = torch.distributions.categorical.Categorical(logits=lprobs)
    samples = distrib.sample([nsamples]).permute([1, 2, 0])
    return samples

def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels)
    return log_likelihood.mean(dim=1)

def get_sampling_discrepancy(logits_ref, logits_score, labels):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    samples = get_samples(logits_ref, labels)
    log_likelihood_x = get_likelihood(logits_score, labels)
    log_likelihood_x_tilde = get_likelihood(logits_score, samples)
    miu_tilde = log_likelihood_x_tilde.mean(dim=-1)
    sigma_tilde = log_likelihood_x_tilde.std(dim=-1)
    discrepancy = (log_likelihood_x.squeeze(-1) - miu_tilde) / sigma_tilde
    return discrepancy.item()

def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
    discrepancy = discrepancy.mean()
    return discrepancy.item()

# estimate the probability according to the distribution of our test results on ChatGPT and GPT-4
class ProbEstimator:
    def __init__(self, args):
        self.real_crits = []
        self.fake_crits = []
        for result_file in glob.glob(os.path.join(args.ref_path, '*.json')):
            with open(result_file, 'r') as fin:
                res = json.load(fin)
                self.real_crits.extend(res['predictions']['real'])
                self.fake_crits.extend(res['predictions']['samples'])
        print(f'ProbEstimator: total {len(self.real_crits) * 2} samples.')


    def crit_to_prob(self, crit):
        offset = np.sort(np.abs(np.array(self.real_crits + self.fake_crits) - crit))[100]
        cnt_real = np.sum((np.array(self.real_crits) > crit - offset) & (np.array(self.real_crits) < crit + offset))
        cnt_fake = np.sum((np.array(self.fake_crits) > crit - offset) & (np.array(self.fake_crits) < crit + offset))
        return cnt_fake / (cnt_real + cnt_fake)


def load_test_dataset(dataset_path):

    dataset = load_from_disk(dataset_path)
    dataset_test = dataset["test"]

    return dataset_test

def tokenize_dataset(tokenizer, dataset):

    if tokenizer is None:
        raise ValueError("Tokenizer not set")
    
    def tokenize_text(x, tokenizer):
        return tokenizer(x["text"], truncation=True, padding="max_length", return_tensors="pt")
    
    dataset = dataset.map(lambda x: tokenize_text(x, tokenizer), batched=True)

    return dataset

   def sample_from_model(model, tokenizer, texts, min_words=55, prompt_tokens=30):
        # encode each text as a list of token ids
        device = model.device
        all_encoded = tokenizer(texts, return_tensors="pt", padding=True, return_token_type_ids=False).to(device)
        all_encoded = {key: value[:, :prompt_tokens] for key, value in all_encoded.items()}

        model.eval()
        decoded = ['' for _ in range(len(texts))]

        # sample from the model until we get a sample with at least min_words words for each example
        # this is an inefficient way to do this (since we regenerate for all inputs if just one is too short), but it works
        tries = 0
        m = 0

        min_chars = 500
        while m < min_chars:
            if tries != 0:
                print()
                print(f"min words: {m}, needed {min_words}, regenerating (try {tries})")
                prefixes = tokenizer.batch_decode(all_encoded['input_ids'], skip_special_tokens=True)
                for prefix, x in zip(prefixes, decoded):
                    if len(x.split()) == m:
                        print(prefix, '=>', x)

            sampling_kwargs = {}
            sampling_kwargs['top_p'] = 0.96
            sampling_kwargs['temperature'] = 0.8
            min_length = 50 if self.args.dataset in ['pubmed'] else 150
            outputs = model.generate(**all_encoded, min_length=min_length, max_length=200, do_sample=True,
                                                **sampling_kwargs, pad_token_id=tokenizer.eos_token_id,
                                                eos_token_id=tokenizer.eos_token_id)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            m = min(len(x) for x in decoded)
            tries += 1

        return decoded

def generate_samples(raw_data, batch_size, model, tokenizer):
    # trim to shorter length
    def _trim_to_shorter_length(texta, textb):
        # truncate to shorter of o and s
        shorter_length = min(len(texta.split(' ')), len(textb.split(' ')))
        texta = ' '.join(texta.split(' ')[:shorter_length])
        textb = ' '.join(textb.split(' ')[:shorter_length])
        return texta, textb

    def _truncate_to_substring(text, substring, idx_occurrence):
        # truncate everything after the idx_occurrence occurrence of substring
        assert idx_occurrence > 0, 'idx_occurrence must be > 0'
        idx = -1
        for _ in range(idx_occurrence):
            idx = text.find(substring, idx + 1)
            if idx == -1:
                return text
        return text[:idx]

    data = {
        "original": [],
        "sampled": [],
    }

    for batch in range(len(raw_data) // batch_size):
        print('Generating samples for batch', batch, 'of', len(raw_data) // batch_size)
        original_text = raw_data[batch * batch_size:(batch + 1) * batch_size]
        sampled_text = sample_from_model(model, tokenizer, original_text, min_words=55)

        for o, s in zip(original_text, sampled_text):

            o, s = _trim_to_shorter_length(o, s)

            # add to the data
            data["original"].append(o)
            data["sampled"].append(s)

    return data
### MAIN FILE ###
def run(args):
    # load model
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.dataset, args.cache_dir)
    scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
    scoring_model.eval()
    if args.reference_model_name != args.scoring_model_name:
        reference_tokenizer = load_tokenizer(args.reference_model_name, args.dataset, args.cache_dir)
        reference_model = load_model(args.reference_model_name, args.device, args.cache_dir)
        reference_model.eval()
    # evaluate criterion
    name = "sampling_discrepancy_analytic"
    criterion_fn = get_sampling_discrepancy_analytic
    prob_estimator = ProbEstimator(args)

    dataset = load_test_dataset(args.dataset_path)

    # maybe need to tokenize the dataset twice, once for each tokenizer
    if args.reference_model_name != args.scoring_model_name:
        pass
    else:
        tokenized_dataset = tokenize_dataset(scoring_tokenizer, dataset)

        

    dataset = list(dataset["test"]["text"])
    labels = list(dataset["test"]["label"])

    dataset = dataset[:10]
    tokenized_data_ref = reference_tokenizer(dataset, truncation=True, padding="max_length", return_tensors="pt")
    data = [x for x, y in zip(data, tokenized_data_ref["input_ids"]) if len(y) <= 512]
    
    n_samples = 1000
    data_sampled = generate_samples(data[:n_samples], batch_size=args.batch_size)

    # save data_sampled to json

    # produce sampled version of the dataset by taking the first 30 tokens of each text and continuing it with the reference model
    #dataset_sampled = produce_sampled_dataset(dataset, reference_model, reference_tokenizer, args.device)


    # iterate over the dataset and do detection on each sample

    # create dataloader
    batch_size = 2

    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Performing detection on dataset..."):
            #text = tokenized_dataset[i]["text"]
            #labels = tokenized_dataset[i]["input_ids"][:, 1:]
            #tokenized = tokenized_dataset[i].to(args.device)
            text = batch["text"]
            #labels = batch["input_ids"][:, 1:]
            tokenized = scoring_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
            labels = tokenized.input_ids[:, 1:]

            
            logits_score = scoring_model(**tokenized).logits[:, :-1]
            if args.reference_model_name == args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = reference_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = reference_model(**tokenized).logits[:, :-1]

            for i in range(batch_size):
                crit = criterion_fn(logits_ref[i:i+1], logits_score[i:i+1], labels[i:i+1])
                #sampled_crit = ...

                # decision = sampled_crit > orig_crit
                # estimate the probability of machine generated text

                #print("crit: ", crit)
                prob = prob_estimator.crit_to_prob(crit)
                #print(f'Fast-DetectGPT criterion is {crit:.4f}, suggesting that the text has a probability of {prob * 100:.0f}% to be fake.')
                #print()
                pred = 1 if prob > 0.5 else 0
                preds.append(pred)

            
            #crit = criterion_fn(logits_ref, logits_score, labels)
            ## estimate the probability of machine generated text
            #prob = prob_estimator.crit_to_prob(crit)
            #print(f'Fast-DetectGPT criterion is {crit:.4f}, suggesting that the text has a probability of {prob * 100:.0f}% to be fake.')
            #print()

    # calculate accuracy
    preds = np.array(preds)
    labels = np.array(dataset["label"])
    acc = np.mean(preds == labels)
    print(f'Accuracy: {acc * 100:.2f}%')

    # results for random prediction
    random_preds = np.random.randint(0, 2, len(labels))
    random_acc = np.mean(random_preds == labels)
    print(f'Random prediction accuracy: {random_acc * 100:.2f}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_model_name', type=str, default="gpt-neo-2.7B")  # use gpt-j-6B for more accurate detection
    parser.add_argument('--scoring_model_name', type=str, default="gpt-neo-2.7B")
    parser.add_argument('--dataset_path', type=str, default="xsum")
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--ref_path', type=str, default="./local_infer_ref")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    run(args)



