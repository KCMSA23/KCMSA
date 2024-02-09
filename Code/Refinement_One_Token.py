
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from transformers import AutoTokenizer, BertForMaskedLM
import A_load_data
from sklearn.preprocessing import OneHotEncoder

device = torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained("KLUE_BERT")
positive_words, neutral_words, negative_words = [], [], []

with open("KCMSA_dataset/Positive_word_list.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip("\n")
        tokenized_text = tokenizer.tokenize(line)
        tokenids = tokenizer.convert_tokens_to_ids(tokenized_text)
        
        if len(tokenids) > 1:
          continue
        flag = False
        for tokenid in tokenids:
            if tokenid < 5:
                flag = True
        if flag:
            continue
        positive_words.append(line)
    
with open("KCMSA_dataset/Neutral_word_list.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip("\n")
        tokenized_text = tokenizer.tokenize(line)
        tokenids = tokenizer.convert_tokens_to_ids(tokenized_text)
        
        if len(tokenids) > 1:
          continue

        flag = False
        for tokenid in tokenids:
            if tokenid < 5:
                flag = True
        if flag:
            continue
        
        neutral_words.append(line)
        
with open("KCMSA_dataset/Negative_word_list.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip("\n")
        tokenized_text = tokenizer.tokenize(line)
        tokenids = tokenizer.convert_tokens_to_ids(tokenized_text)
        
        if len(tokenids) > 1:
          continue
        flag = False
        for tokenid in tokenids:
            if tokenid < 5:
                flag = True
        if flag:
            continue
        
        negative_words.append(line)

with open("KCMSA_dataset/Positve_word_list_one.txt", "w") as f:
  for positve_word in positive_words:
      f.write(positve_word + "\n")
with open("KCMSA_dataset/Neutral_word_list_one.txt", "w") as f:
  for neutral_word in neutral_words:
      f.write(neutral_word + "\n")
with open("KCMSA_dataset/Negative_word_list_one.txt", "w") as f:
  for negative_word in negative_words:
      f.write(negative_word + "\n")
