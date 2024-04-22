import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from generator import LLMGenerator



default_gen_params = {
    "max_length": 200,
    "max_new_tokens": None,
    "temperature": 0.8,
    "top_p": 0.8,
    "repetition_penalty": 1,
    "do_sample": True
}

def load_generator(model_name, device, access_token=None):
    """
    Load the generator model and tokenizer
    """
    # load generator
    if model_name == "qwen_chat":
        gen_path = "Qwen/Qwen1.5-0.5B-Chat"
        gen_model = AutoModelForCausalLM.from_pretrained(gen_path, torch_dtype="auto").to(device)
        gen_tokenizer = AutoTokenizer.from_pretrained(gen_path, trust_remote_code=True, padding_side="left")
        generator = LLMGenerator(gen_model, gen_tokenizer, gen_params=default_gen_params)

        #template for chat
        use_chat_template = True
        template_type ="system_user"

    elif model_name == "qwen_0.5b":
        gen_path = "Qwen/Qwen1.5-0.5B"
        gen_model = AutoModelForCausalLM.from_pretrained(gen_path, torch_dtype="auto").to(device)
        gen_tokenizer = AutoTokenizer.from_pretrained(gen_path, trust_remote_code=True, padding_side="left")
        generator = LLMGenerator(gen_model, gen_tokenizer, gen_params=default_gen_params)

        #template for chat
        use_chat_template = False
        template_type = None        

    elif model_name == "gpt2":
        gen_path = "openai-community/gpt2"
        gen_model = AutoModelForCausalLM.from_pretrained(gen_path, torch_dtype="auto").to(device)
        gen_tokenizer = AutoTokenizer.from_pretrained(gen_path, trust_remote_code=True, padding_side="left")

        gen_params = default_gen_params
        gen_params["repetition_penalty"] = 2.0
        
        # special for gpt2
        gen_tokenizer.pad_token = gen_tokenizer.eos_token
        gen_tokenizer.padding_side = 'left'

        generator = LLMGenerator(gen_model, gen_tokenizer, gen_params=default_gen_params)

        #template for chat
        use_chat_template = False
        template_type = None
    elif model_name == "gemma_2b_chat":
        gen_path = "google/gemma-2b-it"
        gen_model = AutoModelForCausalLM.from_pretrained(gen_path,  token=access_token, torch_dtype=torch.bfloat16).to(device)
        gen_tokenizer = AutoTokenizer.from_pretrained(gen_path,  token=access_token)
        generator = LLMGenerator(gen_model, gen_tokenizer, device, gen_params=default_gen_params)

        #template for chat
        use_chat_template = True
        template_type ="user"

    elif model_name == "gemma_2b":
        gen_path = "google/gemma-2b"
        gen_model = AutoModelForCausalLM.from_pretrained(gen_path,  token=access_token, torch_dtype=torch.bfloat16).to(device)
        gen_tokenizer = AutoTokenizer.from_pretrained(gen_path,  token=access_token)
        generator = LLMGenerator(gen_model, gen_tokenizer, device, gen_params=default_gen_params)

        #template for chat
        use_chat_template = False
        template_type = None  

    elif model_name == "phi":
        gen_path = "microsoft/phi-2"
        gen_model = AutoModelForCausalLM.from_pretrained(gen_path, torch_dtype=torch.float16).to(device)
        gen_tokenizer = AutoTokenizer.from_pretrained(gen_path)
        generator = LLMGenerator(gen_model, gen_tokenizer, device, gen_params=default_gen_params)

        # special for phi
        gen_tokenizer.pad_token = gen_tokenizer.eos_token
        gen_tokenizer.padding_side = 'left'

        #template for chat
        use_chat_template = False
        template_type = None  

    elif model_name == "mistral":
        gen_path = "mistralai/Mistral-7B-v0.1"
        gen_model = AutoModelForCausalLM.from_pretrained(gen_path, torch_dtype=torch.bfloat16).to(device)
        gen_tokenizer = AutoTokenizer.from_pretrained(gen_path, trust_remote_code=True)
        generator = LLMGenerator(gen_model, gen_tokenizer, gen_params=default_gen_params)

        # special for mistral
        gen_tokenizer.pad_token = gen_tokenizer.eos_token

        #template for chat
        use_chat_template = False
        template_type = None  

    elif model_name == "zephyr":
        gen_path = "HuggingFaceH4/zephyr-7b-beta"
        gen_model = AutoModelForCausalLM.from_pretrained(gen_path, torch_dtype=torch.bfloat16).to(device)
        gen_tokenizer = AutoTokenizer.from_pretrained(gen_path, trust_remote_code=True)
        generator = LLMGenerator(gen_model, gen_tokenizer, gen_params=default_gen_params)

        # special for mistral
        gen_tokenizer.pad_token = gen_tokenizer.eos_token

        #template for chat
        use_chat_template = True
        template_type ="user"

    else:
        # no other generator is supported for now
        raise ValueError("Generator not supported")
    
    return generator, gen_tokenizer, use_chat_template, template_type