import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F
import transformers
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import A_load_data
import random 

import argparse 
parser = argparse.ArgumentParser(description="models setting")
parser.add_argument('--learning_rate', dest='learning_rate', type=float, help='learning_rate', default=5e-5)
parser.add_argument('--epoch', dest='epoch', type=int, help='epoch', default=4)
parser.add_argument('--iter', dest='iter', type=int, help='iter', default=0)

args = parser.parse_args()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
randomseed = 3407
batch_size = 64
hidden_size = 768
n_class = 3
maxlen = 100
bert_model = "KLUE_BERT"

epoches = args.epoch


class BertDataset(Data.Dataset):
  def __init__(self, sentences, labels=None, with_labels=True, ):
    self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
    self.with_labels = with_labels
    self.sentences = sentences
    self.labels = labels
  def __len__(self):
    return len(self.sentences)

  def __getitem__(self, index):
    # Selecting sentence1 and sentence2 at the specified index in the data frame
    sent = self.sentences[index]

    # Tokenize the pair of sentences to get token ids, attention masks and token type ids
    encoded_pair = self.tokenizer(sent,
                    padding='max_length',  # Pad to max_length
                    truncation=True,       # Truncate to max_length
                    max_length=maxlen,  
                    return_tensors='pt')  # Return torch.Tensor objects

    token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
    attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
    token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

    if self.with_labels:  # True if the dataset has labels
      label = self.labels[index]
      return token_ids, attn_masks, token_type_ids, label
    else:
      return token_ids, attn_masks, token_type_ids

class BertClassifier(nn.Module):
  def __init__(self, hidden_size, n_class,):
    super(BertClassifier, self).__init__()
    self.bert = AutoModel.from_pretrained(bert_model, output_hidden_states=True, return_dict=True)
    for param in self.bert.parameters():
        param.requires_grad = True
    self.linear = nn.Linear(hidden_size, n_class) # 直接用cls向量接全连接层分类
    self.dropout = nn.Dropout(0.5)

  def forward(self, X):
    input_ids, attention_mask, token_type_ids = X[0], X[1], X[2]
    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids) # 返回一个output字典
    # 用最后一层cls向量做分类
    # outputs.pooler_output: [bs, hidden_size]
    outputs = self.linear(self.dropout(outputs.pooler_output))
    return outputs

if __name__ == '__main__':
  torch.cuda.manual_seed(randomseed + args.iter)
  torch.manual_seed(randomseed + args.iter)
  np.random.seed(randomseed + args.iter)
  random.seed(randomseed + args.iter)


  X_train, Y_train = A_load_data.load_twitter_train()
  X_test, Y_test = A_load_data.load_twitter_test()


  encoder = OneHotEncoder(categories="auto", sparse=False)
  Y_train = encoder.fit_transform(np.expand_dims(np.array(Y_train), axis=-1))
  Y_train = torch.from_numpy(Y_train)
  Y_test = encoder.fit_transform(np.expand_dims(np.array(Y_test), axis=-1))
  Y_test = torch.from_numpy(Y_test)
  train = Data.DataLoader(dataset=BertDataset(X_train, Y_train), batch_size=batch_size, shuffle=True, num_workers=1)
  test = Data.DataLoader(dataset=BertDataset(X_test, Y_test), batch_size=batch_size, shuffle=False, num_workers=1)


  bc = BertClassifier(hidden_size, n_class).to(device)
  optimizer = optim.Adam(bc.parameters(), lr=args.learning_rate)
  loss_fn = nn.CrossEntropyLoss()   

  # train
  bc.train()
  for epoch in range(epoches):
      loss_list = []
      for batch in train:
          optimizer.zero_grad()
          batch = tuple(p.to(device) for p in batch)
          pred = bc([batch[0], batch[1], batch[2]])
          loss = F.cross_entropy(pred, batch[3].argmax(dim=-1).to(device), reduction="mean")
          loss_list.append(loss.item())
          loss.backward()
          optimizer.step()
      print('[{}|{}]  loss:{:.4f}'.format(epoch+1, epoches, np.mean(np.array(loss_list))))

  bc.eval()
  true_list, preds_list = [], []
  for batch in test:
      with torch.no_grad():
          batch = tuple(p.to(device) for p in batch)
          true_list.append(batch[3].detach().cpu().numpy())
          preds = bc([batch[0], batch[1], batch[2]])
          preds = torch.softmax(preds, dim=-1)
          preds_list.append(preds.detach().cpu().numpy())
  true_np, preds_np = np.concatenate(true_list), np.concatenate(preds_list)
  y_true = np.argmax(true_np, axis=-1)
  y_pred = np.argmax(preds_np, axis=-1)
  acc = accuracy_score(y_true, y_pred)
  f1 = f1_score(y_true, y_pred, average="macro")

  print(acc)
  print(f1)

  file1 = open("result_acc.txt","a")
  file2 = open("result_f1.txt","a")

  print(acc, file=file1)
  print(f1, file=file2)