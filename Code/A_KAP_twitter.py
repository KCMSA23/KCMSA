import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F
import transformers
from transformers import AutoModel, AutoTokenizer, BertForMaskedLM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import A_load_data
import argparse 
import random

parser = argparse.ArgumentParser(description="models setting")
parser.add_argument('--learning_rate', dest='learning_rate', type=float, help='learning_rate', default=5e-5)
parser.add_argument('--epoch', dest='epoch', type=int, help='epoch', default=5)
parser.add_argument('--iter', dest='iter', type=int, help='iter', default=0)
parser.add_argument('--template_id', dest='template_id', type=int, help='template_id', default=4)
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

batch_size = 64
randomseed = 3407
maxlen = 160
hidden_size = 768
template_id = args.template_id
epoches = args.epoch


bert_model = "KLUE_BERT"
tokenizer = AutoTokenizer.from_pretrained(bert_model)
vocab_size = tokenizer.vocab_size
n_class = vocab_size

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
    if template_id == 1:
        sent = "그것은 [MASK]적이 였다. " + self.sentences[index]
    if template_id == 2:
      sent = "그냥 [MASK]적이 였다. " + self.sentences[index]
    if template_id == 3:
      sent = self.sentences[index] + " 전적으로 [MASK]적이 였다."
    if template_id == 4:
      sent = self.sentences[index] + " 요약하자면 이 문자는 [MASK]적이 였다." 
    if template_id == 5:
      sent = self.sentences[index] + " 생각해보면 [MASK]적이 였다." 
    if template_id == 6:
      sent = self.sentences[index] + " 한마디로 [MASK]적이 였다." 
    if template_id == 7:
      sent = self.sentences[index] + " 본 단락의 내용을 요약하자면 [MASK]적이 였다." 
    if template_id == 8:
      sent = self.sentences[index] + " 그래서 [MASK]적이라고 생각한다." 
    if template_id == 9:
      sent = self.sentences[index] + " 그렇지만 [MASK]적이 였다." 
    if template_id == 10:
      sent = self.sentences[index] + " 이에 관하여 나는 [MASK]적이라고 생각한다." 

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
  def __init__(self, postive_tokenids, neutral_tokenids, negative_tokenids):
    super(BertClassifier, self).__init__()
    self.bert = BertForMaskedLM.from_pretrained(bert_model, output_hidden_states=True, return_dict=True)
    for param in self.bert.parameters():
        param.requires_grad = True
    self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
    self.wpostive = nn.Parameter(torch.FloatTensor(torch.zeros((len(postive_tokenids)))).to(device), requires_grad=True)
    self.wneutral = nn.Parameter(torch.FloatTensor(torch.zeros((len(neutral_tokenids)))).to(device), requires_grad=True)
    self.wnegative = nn.Parameter(torch.FloatTensor(torch.zeros((len(negative_tokenids)))).to(device), requires_grad=True)

    self.postive_tokenids = postive_tokenids
    self.neutral_tokenids = neutral_tokenids
    self.negative_tokenids = negative_tokenids

  def forward(self, X):
    input_ids, attention_mask, token_type_ids = X[0], X[1], X[2]
    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    preds = outputs.logits[torch.where(input_ids == tokenizer.mask_token_id)]
    preds = torch.softmax(preds, dim=-1)   

    alphapositive = torch.exp(self.wpostive) / torch.sum(torch.exp(self.wpostive))
    alphaneutral = torch.exp(self.wneutral) / torch.sum(torch.exp(self.wneutral))
    alphanegative = torch.exp(self.wnegative) / torch.sum(torch.exp(self.wnegative))
    
    pro_positve = torch.FloatTensor(torch.zeros((preds.shape[0], len(postive_tokenids)))).to(device)
    for i in range(len(self.postive_tokenids)):
      tokenids = self.postive_tokenids[i]
      prob = torch.FloatTensor(torch.zeros((preds.shape[0]))).to(device)
      for tokenid in tokenids:
        prob += preds[:, tokenid]
      prob = prob / len(tokenids)
      pro_positve[:, i] += prob
    s_positive = torch.log(pro_positve) @ alphapositive

    pro_neutral = torch.FloatTensor(torch.zeros((preds.shape[0], len(neutral_tokenids)))).to(device)
    for i in range(len(self.neutral_tokenids)):
      tokenids = self.neutral_tokenids[i]
      prob = torch.FloatTensor(torch.zeros((preds.shape[0]))).to(device)
      for tokenid in tokenids:
        prob += preds[:, tokenid]
      prob = prob / len(tokenids)
      pro_neutral[:, i] += prob
    s_neutral = torch.log(pro_neutral) @ alphaneutral

    pro_negative = torch.FloatTensor(torch.zeros((preds.shape[0], len(negative_tokenids)))).to(device)
    for i in range(len(self.negative_tokenids)):
      tokenids = self.negative_tokenids[i]
      prob = torch.FloatTensor(torch.zeros((preds.shape[0]))).to(device)
      for tokenid in tokenids:
        prob += preds[:, tokenid]
      prob = prob / len(tokenids)
      pro_negative[:, i] += prob
    s_negative = torch.log(pro_negative) @ alphanegative

    return torch.cat([s_neutral.unsqueeze(1), s_positive.unsqueeze(1), s_negative.unsqueeze(1)], dim=1)

def get_word_tokens(postive_words,neutral_words,negative_words):
  postive_tokenids = []
  for positve_word in postive_words:
    tokenized_text = tokenizer.tokenize(positve_word)
    tokenids = tokenizer.convert_tokens_to_ids(tokenized_text)
    postive_tokenids.append(tokenids)
  neutral_tokenids = []
  for neutral_word in neutral_words:
    tokenized_text = tokenizer.tokenize(neutral_word)
    tokenids = tokenizer.convert_tokens_to_ids(tokenized_text)
    neutral_tokenids.append(tokenids)
  negative_tokenids = []
  for negative_word in negative_words:
    tokenized_text = tokenizer.tokenize(negative_word)
    tokenids = tokenizer.convert_tokens_to_ids(tokenized_text)
    negative_tokenids.append(tokenids)
  return postive_tokenids, neutral_tokenids, negative_tokenids


if __name__ == '__main__':
  torch.cuda.manual_seed(randomseed + args.iter)
  torch.manual_seed(randomseed + args.iter)
  np.random.seed(randomseed + args.iter)
  random.seed(randomseed + args.iter)
 
  X_train, Y_train = A_load_data.load_twitter_train()
  X_test, Y_test = A_load_data.load_twitter_test()
  

  postive_words,neutral_words,negative_words = A_load_data.get_label_words(template_id)
  postive_tokenids, neutral_tokenids, negative_tokenids = get_word_tokens(postive_words,neutral_words,negative_words)

  encoder = OneHotEncoder(categories="auto", sparse=False)
  Y_train = encoder.fit_transform(np.expand_dims(np.array(Y_train), axis=-1))
  Y_train = torch.from_numpy(Y_train)
  Y_test = encoder.fit_transform(np.expand_dims(np.array(Y_test), axis=-1))
  Y_test = torch.from_numpy(Y_test)
  train = Data.DataLoader(dataset = BertDataset(X_train, Y_train), batch_size=batch_size, shuffle=True, num_workers=1)
  test = Data.DataLoader(dataset = BertDataset(X_test, Y_test), batch_size=1, shuffle=False, num_workers=1)


  bc = BertClassifier(postive_tokenids, neutral_tokenids, negative_tokenids).to(device)
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
  
  print(acc, f1)

  file1 = open("result_acc.txt","a")
  file2 = open("result_f1.txt","a")

  print(acc, file=file1)
  print(f1, file=file2)
