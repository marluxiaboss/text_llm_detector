import torch
import torch.nn as nn

class LLMGenerator(nn.Module):
  def __init__(self, gpt_model, tokenizer):
    super().__init__()

    # gpt should already be trained
    self.gpt = gpt_model
    self.tokenizer = tokenizer
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  def forward_old(self, text, max_length=512, max_new_tokens=100, temperature=1, top_k=50, top_p=0.9, repetition_penalty=1, skip_special_tokens=True):

    # tokenize text using the tokenizer
    input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
    # generate text using the gpt model
    #output_ids = self.gpt.generate(input_ids, max_length=max_length, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty)
    with torch.no_grad():
      output_ids = self.gpt.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty)

    # optional, remove input_ids from output_ids
    #output_ids = [output_id[len(input_ids):] for input_id, output_id in zip(input_ids, output_ids)]

    # decode the generated text
    decoded_output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=skip_special_tokens)[0]

    # remove the input text from the generated text
    decoded_output = decoded_output.replace(text, "").strip()

    # remove special tokens from the generated text
    special_tokens = self.tokenizer.additional_special_tokens

    for special_token in special_tokens:
      decoded_output = decoded_output.replace(special_token, "")

    #decoded_output = decoded_output.replace(text, "")
    return decoded_output
  

  def forward(self, samples, max_length=512, max_new_tokens=100, temperature=1, top_k=50, top_p=0.9, repetition_penalty=1, skip_special_tokens=True):

    encoding = self.tokenizer.batch_encode_plus(samples, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    input_ids = encoding['input_ids'].to(self.device)

    # generate text using the gpt model
    with torch.no_grad():
        output_ids = self.gpt.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty)

    # decode the generated text
    #decoded_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=skip_special_tokens)
    decoded_outputs = self.tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:])
        

    # remove special tokens from the generated text
    special_tokens = self.tokenizer.additional_special_tokens + [self.tokenizer.pad_token] + [self.tokenizer.eos_token]
    for i, sample in enumerate(samples):
        decoded_output = decoded_outputs[i]
        for special_token in special_tokens:
            decoded_output = decoded_output.replace(special_token, "")
        decoded_outputs[i] = decoded_output

    return decoded_outputs


