import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class LLMDetector(nn.Module):
  def __init__(self, bert_model, tokenizer, num_classes):
    super().__init__()

    self.tokenizer = tokenizer

    # bert should already be trained
    self.bert = bert_model
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set num_classes
    self.num_classes = num_classes

  def forward(self, text):

    # tokenize text using the tokenizer
    output = self.tokenizer(text, return_tensors="pt")
    input_ids = output["input_ids"].to(self.device)
    logits = self.bert(input_ids)["logits"]

    # apply sigmoid to get probabilities of each class
    output = torch.sigmoid(logits)
    return output