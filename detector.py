import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class LLMDetector(nn.Module):
  def __init__(self, bert_model, tokenizer, num_classes, freeze_bert=False, device=None):
    super().__init__()

    self.tokenizer = tokenizer

    # bert should already be trained
    self.bert = bert_model
    
    if device is None:
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
      self.device = device

    # set num_classes
    self.num_classes = num_classes

    if freeze_bert:
      for name, param in self.bert.named_parameters():
        if not name.startswith("classifier"):
          param.requires_grad = False

  def forward(self, text):

    # tokenize text using the tokenizer
    output = self.tokenizer(text, return_tensors="pt")
    input_ids = output["input_ids"].to(self.device)
    logits = self.bert(input_ids)["logits"]

    # apply sigmoid to get probabilities of each class
    output = torch.sigmoid(logits)
    return output
  
  @staticmethod
  def freeze_base(bert_model):
    for name, param in bert_model.named_parameters():
      if not name.startswith("classifier") and not name.startswith("classification_head"):
        param.requires_grad = False