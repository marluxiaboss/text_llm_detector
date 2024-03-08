from datasets import load_dataset, load_from_disk, Dataset, DatasetDict, concatenate_datasets
import numpy as np
import pandas
import argparse
from tqdm import tqdm
import os
import torch

from transformers import (AutoModelForCausalLM, AutoTokenizer, BertForSequenceClassification, BertTokenizer, BertModel,
 RobertaForSequenceClassification, RobertaTokenizer, RobertaModel, TrainingArguments, Trainer)

from generator import LLMGenerator


def create_random_subset(dataset, n=10, seed=42):
    """
    Create a random subset of the dataset
    """
    if n > len(dataset):
        n = len(dataset)
    np.random.seed(seed)
    indices = np.random.choice(len(dataset), n, replace=False)
    subset = dataset.select(indices)
    return subset

def process_true_dataset(true_dataset, fake_dataset_size, seed=42):
    #dataset = load_dataset(dataset_path)

    true_dataset = true_dataset.select_columns(["response", "instruction", "context"])
    true_dataset = true_dataset.rename_column("response", "text")

    # select random samples from true_dataset to match fake_dataset size
    true_dataset = true_dataset.shuffle(seed=seed)
    #true_dataset = true_dataset.select(range(len(fake_dataset["train"])))
    #true_dataset = true_dataset.select(range(fake_dataset_size))
    true_dataset["train"] = true_dataset["train"].select(range(fake_dataset_size))

    # create label = 0 for true responses and label = 1 for fake responses
    true_dataset = true_dataset.map(lambda x: {"label": 0})

    # save dataset
    #true_dataset.save_to_disk("true_dataset")

    return true_dataset


def generate_fake_responses(generator, true_dataset, gen_tokenizer, max_new_tokens, batch_size=2, use_chat_template=False, template_type=None):
    """
    Traverse dataset and generate responses for each instruction
    """

    fake_responses = []
    """
    # TODO: improve this loop by parallelizing/batch
    for data in tqdm(true_dataset, desc="Generating fake responses"):
        # Create query in the format that the generator expects
        text_instruction = f"Context: {data['context']} \n {data['instruction']}"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{text_instruction}"},
        ]
        text = gen_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Generate response
        output = generator(text, skip_special_tokens=False, max_new_tokens=max_new_tokens)
        
        fake_responses.append(output)
    """

    # batch generation
    batch_size = batch_size
    # transform into chat template
    def transform_chat_template(sample, use_chat_template=False):

        text_instruction = f"Context: {sample['context']} \n {sample['instruction']}"
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

            text_template = gen_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            text_template = text_instruction
        return {"text_template": text_template}
    
    true_dataset = true_dataset.map(lambda x: transform_chat_template(x, use_chat_template=use_chat_template))
    true_dataset_list = true_dataset["text_template"]
    
    for i in tqdm(range(0, len(true_dataset_list), batch_size), desc="Generating fake responses"):
        batch = true_dataset_list[i:i+batch_size]
        responses = generator(batch, max_new_tokens=max_new_tokens, skip_special_tokens=True)
        fake_responses.extend(responses)
    return fake_responses

def filter_instruction(sample):
    """
    Filter out the instruction from the generated response
    Note: only works if special tokens are not removed
    """
    """
    text_instruction = f"Context: {sample['context']} \n {sample['instruction']}"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{text_instruction}"},
    ]
    text_template = gen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    generated_response = sample["generated_response"]

    response_without_instruction = generated_response.replace(text_template, "")

    """
    def remove_string_between(str, str_beg, str_end):
        return str.split(str_beg)[0] + str_end + str.split(str_end)[1]

    # removing everything between system and user (including system and user)
    """
    str_beg = "system"
    str_end = "assistant"
    response_without_instruction = remove_string_between(sample["generated_response"], str_beg, str_end).strip()
    response_without_instruction = sample["generated_response"]
    """

    response_without_instruction = sample["generated_response"]
    # remove newline characters
    response_without_instruction = response_without_instruction.replace("\n", " ")

    return {"generated_response": response_without_instruction}

def generate_fake_dataset(true_dataset, fake_dataset_size, generator, gen_tokenizer, max_nb_tokens_input=100, max_new_tokens=100, seed=42, batch_size=2, use_chat_template=False, template_type=None):
    
    # discard instructions that are more than max_nb_tokens_input tokens
    max_nb_tokens_input = max_nb_tokens_input

    # tokenize the instructions
    true_dataset = true_dataset.map(lambda x: {"tokenized_instruction": gen_tokenizer(x["instruction"])})
    true_dataset = true_dataset.map(lambda x: {"tokenized_context": gen_tokenizer(x["context"])})
    dataset_before_len = len(true_dataset["train"])
    true_dataset = true_dataset.filter(lambda x: len(x["tokenized_instruction"]["input_ids"]) + len(x["tokenized_context"]["input_ids"]) <= max_nb_tokens_input)
    dataset_after_len = len(true_dataset["train"])
    print(f"Percent of data discarded after filtering out input > max_nb_tokens: {100*(1 - dataset_after_len/dataset_before_len):.2f}%")

    subset_size = fake_dataset_size
    train_subset = create_random_subset(true_dataset["train"], n=subset_size, seed=seed)

    
    fake_responses_train = generate_fake_responses(generator, train_subset, gen_tokenizer, max_new_tokens=max_new_tokens, batch_size=batch_size, use_chat_template=use_chat_template, template_type=template_type)

    fake_responses_train = Dataset.from_dict({"generated_response": fake_responses_train, "instruction": train_subset["instruction"],
    "context": train_subset["context"], "true_response": train_subset["response"], "category": train_subset["category"]})

    fake_dataset = DatasetDict()
    fake_dataset["train"] = fake_responses_train

    return fake_dataset

def process_fake_dataset(fake_dataset, gen_tokenizer):

    # filter out instruction from generated_response
    fake_dataset = fake_dataset.map(lambda x: filter_instruction(x))

    fake_dataset = fake_dataset.map(lambda x: {"label": 1})

    # remove true_response from fake_dataset
    fake_dataset = fake_dataset.remove_columns(["true_response"])

    # rename generated_response to response
    fake_dataset = fake_dataset.rename_column("generated_response", "text")
    fake_dataset = fake_dataset.select_columns(["text", "label", "instruction", "context"])

    return fake_dataset

def merge_true_fake_dataset(true_dataset, fake_dataset, seed=42):
    
    merged = concatenate_datasets([true_dataset["train"], fake_dataset["train"]])
    merged_dataset = DatasetDict()
    merged_dataset["train"] = merged

    # shuffle the dataset
    merged_dataset = merged_dataset.shuffle(seed=seed)

    # save merged dataset
    #merged_dataset.save_to_disk("merged_dataset")

    return merged_dataset

def split_merged_dataset(merged_dataset, eval_size=0.1, test_size=0.1):
    """
    from https://discuss.huggingface.co/t/how-to-split-main-dataset-into-train-dev-test-as-datasetdict/1090/6
    """

    merged_dataset_train_test_valid = merged_dataset["train"].train_test_split(test_size=test_size + eval_size)

    test_valid = merged_dataset_train_test_valid['test'].train_test_split(test_size=test_size / (test_size + eval_size))

    merged_dataset = DatasetDict({
    'train': merged_dataset_train_test_valid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})

    print("Train size:", len(merged_dataset['train']))
    print("Eval size:", len(test_valid['train']))
    print("Test size:", len(test_valid['test']))

    return merged_dataset

def format_merged_dataset(merged_dataset):
    """
    Format the text into a template
    """

    def format_text(sample):

        text = sample["text"]
        modified_text = f"Instruction: {sample['context']} \n {sample['instruction']} \n Answer: {text}"

        return {"text": modified_text}
    
    merged_dataset = merged_dataset.map(format_text)
    
    # only keep text and label
    merged_dataset = merged_dataset.select_columns(["text", "label"])

    return merged_dataset

def format_news_dataset(true_dataset, prefix_cutoff=10):
    true_dataset["instuction"] = true_dataset["article"].apply(lambda x: " ".join(x.split()[:prefix_cutoff]))
    true_dataset["context"] = ""
    true_dataset["response"] = true_dataset["article"].apply(lambda x: " ".join(x.split()[prefix_cutoff:]))
    true_dataset["category"] = "news"
    true_dataset = true_dataset.remove_columns(["article"])
    true_dataset = true_dataset.remove_columns(["highlights"])
    return true_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--true_dataset_path", type=str, help="Path to the true dataset (hugginface dataset path)", default="databricks/databricks-dolly-15k")
    parser.add_argument("--fake_dataset_size", type=int, help="Size of the fake dataset", default=10)
    parser.add_argument("--max_nb_tokens_input", type=int, help="Max number of tokens for input", default=100)
    parser.add_argument("--generator", type=str, help="Generator model name between 'qwen', 'phi', 'gemma', 'mistral', 'gpt2'", default="qwen")
    parser.add_argument("--device", type=str, help="Device to use for the generator", default="cuda")
    parser.add_argument("--validation_size", type=float, help="Size of the validation set", default=0.1)
    parser.add_argument("--test_size", type=float, help="Size of the test set", default=0.1)
    parser.add_argument("--max_new_tokens", type=int, help="Max length of the generated response", default=512)
    parser.add_argument("--seed", type=int, help="Seed for random number generator", default=42)
    parser.add_argument("--batch_size", type=int, help="Batch size for generation", default=2)
    parser.add_argument("--experiment_name", type=str, help="Name of the experiment", default="test_experiment")
    parser.add_argument("--access_token", type=str, help="Huggingface access token used for Llama and Gemma", default="")

    args = parser.parse_args()

    # check if the dataset already exists for the given experiment name
    if (os.path.exists(f"true_dataset_{args.experiment_name}")
        or os.path.exists(f"fake_dataset_{args.experiment_name}")
        or os.path.exists(f"fake_true_dataset_{args.experiment_name}")):
        print("Warning: dataset already exists for the given experiment name, do you want to overwrite it?")
        answer = input("yes/no: ")
        if answer != "yes":
            exit()


    # TODO: add checks for test_size and validation_size, max_length and max_nb_tokens_input

    # load generator
    if args.generator == "qwen":
        gen_path = "Qwen/Qwen1.5-0.5B-Chat"
        gen_model = AutoModelForCausalLM.from_pretrained(gen_path, torch_dtype="auto").to(args.device)
        gen_tokenizer = AutoTokenizer.from_pretrained(gen_path, trust_remote_code=True, padding_side="left")
        generator = LLMGenerator(gen_model, gen_tokenizer)

        #template for chat
        use_chat_template = True
        template_type ="system_user"

    elif args.generator == "gpt2":
        gen_path = "openai-community/gpt2"
        gen_model = AutoModelForCausalLM.from_pretrained(gen_path, torch_dtype="auto").to(args.device)
        gen_tokenizer = AutoTokenizer.from_pretrained(gen_path, trust_remote_code=True, padding_side="left")
        generator = LLMGenerator(gen_model, gen_tokenizer)

        # special for gpt2
        gen_tokenizer.pad_token = gen_tokenizer.eos_token
        gen_tokenizer.padding_side = 'left'

        #template for chat
        use_chat_template = False
        template_type = None
    elif args.generator == "gemma_2b":
        gen_path = "google/gemma-2b-it"
        gen_model = AutoModelForCausalLM.from_pretrained(gen_path,  token=args.access_token, torch_dtype=torch.bfloat16).to(args.device)
        gen_tokenizer = AutoTokenizer.from_pretrained(gen_path,  token=args.access_token)
        generator = LLMGenerator(gen_model, gen_tokenizer, args.device)

        #template for chat
        use_chat_template = True
        template_type ="user"
    else:
        # no other generator is supported for now
        raise ValueError("Generator not supported")
    

    if args.true_dataset_path == "databricks/databricks-dolly-15k":
        # load true dataset
        true_dataset = load_dataset(args.true_dataset_path)

    elif args.true_dataset_path == 'cnn_dailymail':
        # load true dataset from disk
        true_dataset = load_dataset(args.true_dataset_path, "3.0.0")

        # format dataset to have the same columns as the other datasets: "instruction", "context", "response", "category"
        true_dataset = format_news_dataset(true_dataset["train"])        

    else:
        raise ValueError("Dataset not supported")

    # generate fake dataset
    fake_dataset = generate_fake_dataset(true_dataset, args.fake_dataset_size, generator, gen_tokenizer, args.max_nb_tokens_input, args.max_new_tokens, args.seed, args.batch_size, use_chat_template=use_chat_template, template_type=template_type)

    # process true dataset
    true_dataset = process_true_dataset(true_dataset, args.fake_dataset_size, args.seed)
    true_dataset.save_to_disk(f"true_dataset_{args.experiment_name}")

    # process fake dataset
    fake_dataset = process_fake_dataset(fake_dataset, gen_tokenizer)
    fake_dataset.save_to_disk(f"fake_dataset_{args.experiment_name}")

    # merge true and fake dataset
    merged_dataset = merge_true_fake_dataset(true_dataset, fake_dataset, args.seed)

    # format merged dataset into a template
    merged_dataset = format_merged_dataset(merged_dataset)

    # split merged dataset into train, eval, test
    merged_dataset = split_merged_dataset(merged_dataset, eval_size=args.validation_size, test_size=args.test_size)
    merged_dataset.save_to_disk(f"fake_true_dataset_{args.experiment_name}")




    

