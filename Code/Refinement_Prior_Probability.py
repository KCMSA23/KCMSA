
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from transformers import AutoTokenizer, BertForMaskedLM
import A_load_data
from sklearn.preprocessing import OneHotEncoder

device = torch.device('cuda:0')
template_id = 1
maxlen = 160

for template_id in range(1, 2):
  tokenizer = AutoTokenizer.from_pretrained("KLUE_BERT")
  positive_words, neutral_words, negative_words = [], [], []
  positive_list, neutral_list, negative_list = [], [], []
  positive_p, neutral_p, negative_p = [], [], []

  with open("KCMSA_dataset/Positive_word_list_one.txt", "r") as f:
      lines = f.readlines()
      for line in lines:
          line = line.strip("\n")
          tokenized_text = tokenizer.tokenize(line)
          tokenids = tokenizer.convert_tokens_to_ids(tokenized_text)
          
          flag = False
          for tokenid in tokenids:
              if tokenid < 5:
                  flag = True
          if flag:
              continue
          positive_words.append(line)
          positive_list.append(tokenids)
          positive_p.append([])
  with open("KCMSA_dataset/Neutral_word_list_one.txt", "r") as f:
      lines = f.readlines()
      for line in lines:
          line = line.strip("\n")
          tokenized_text = tokenizer.tokenize(line)
          tokenids = tokenizer.convert_tokens_to_ids(tokenized_text)
          
          flag = False
          for tokenid in tokenids:
              if tokenid < 5:
                  flag = True
          if flag:
              continue
          
          neutral_words.append(line)
          neutral_list.append(tokenids)
          neutral_p.append([])
  with open("KCMSA_dataset/Negative_word_list_one.txt", "r") as f:
      lines = f.readlines()
      for line in lines:
          line = line.strip("\n")
          tokenized_text = tokenizer.tokenize(line)
          tokenids = tokenizer.convert_tokens_to_ids(tokenized_text)
          
          flag = False
          for tokenid in tokenids:
              if tokenid < 5:
                  flag = True
          if flag:
              continue
          
          negative_words.append(line)
          negative_list.append(tokenids)
          negative_p.append([])


  class BertTestDataset(Data.Dataset):
    def __init__(self, sentences, labels=None, with_labels=True, ):
      self.tokenizer = AutoTokenizer.from_pretrained("KLUE_BERT")
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
    def __init__(self):
      super(BertClassifier, self).__init__()
      self.bert = BertForMaskedLM.from_pretrained("KLUE_BERT", output_hidden_states=True, return_dict=True)
      for param in self.bert.parameters():
          param.requires_grad = True
      self.tokenizer = AutoTokenizer.from_pretrained("KLUE_BERT")

    def forward(self, X, label=None):
      input_ids, attention_mask, token_type_ids = X[0], X[1], X[2]
      if label == None:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
      else:
        labels = torch.where(input_ids == tokenizer.mask_token_id, label, -100)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels = labels)
      return outputs

  X_train, Y_train = A_load_data.load_weibo_train()
  encoder = OneHotEncoder(categories="auto", sparse=False)
  Y_train = encoder.fit_transform(np.expand_dims(np.array(Y_train), axis=-1))
  Y_train = torch.from_numpy(Y_train)
  test = Data.DataLoader(dataset=BertTestDataset(X_train, Y_train), batch_size=1, shuffle=False, num_workers=1)    

  bc = BertClassifier().to(device)
  bc.eval()
  iter = 0
  for batch in test:
      if iter % 100 == 0:
          print(iter)
      iter += 1 
      with torch.no_grad():
          batch = tuple(p.to(device) for p in batch)
          
          #tokenid==4: [mask]
          pos = torch.where(batch[0] == 4)
          preds = bc([batch[0], batch[1], batch[2]]).logits[pos]
          preds = torch.softmax(preds, dim=-1)    
          for i in range(len(positive_list)):
              word_ids = positive_list[i]
              prob = []
              for wordid in word_ids:
                  prob.append(preds[:, wordid].item())
              prob = np.array(prob)
              positive_p[i].append(np.mean(prob))
          for i in range(len(neutral_list)):
              word_ids = neutral_list[i]
              prob = []
              for wordid in word_ids:
                  prob.append(preds[:, wordid].item())
              prob = np.array(prob)
              neutral_p[i].append(np.mean(prob))
          for i in range(len(negative_list)):
              word_ids = negative_list[i]
              prob = []
              for wordid in word_ids:
                  prob.append(preds[:, wordid].item())
              prob = np.array(prob)
              negative_p[i].append(np.mean(prob))


  for i in range(len(positive_p)):
    print(len(positive_p[i]))
    p_i = np.mean(np.array(positive_p[i]))
    positive_p[i] = p_i
  for i in range(len(neutral_p)):
    p_i = np.mean(np.array(neutral_p[i]))
    neutral_p[i] = p_i
  for i in range(len(negative_p)):
    p_i = np.mean(np.array(negative_p[i]))
    negative_p[i] = p_i

  positive_FR = dict(zip(positive_words,positive_p))
  positive_FR = sorted(positive_FR, reverse=True)[:100]
  neutral_FR = dict(zip(neutral_words,neutral_p))
  neutral_FR = sorted(neutral_FR, reverse=True)
  negative_FR = dict(zip(negative_words,negative_p))
  negative_FR = sorted(negative_FR, reverse=True)[:100]

  positive_p_t = sorted(positive_p, reverse=True)
  temp = positive_p_t[:10]
  for item in temp:
     print(item)
  print("*"* 100)

  neutral_p_t = sorted(neutral_p, reverse=True)
  temp = neutral_p_t[:10]
  for item in temp:
     print(item)
  print("*"* 100)
  negative_p_t = sorted(negative_p, reverse=True)
  temp = negative_p_t[:10]
  for item in temp:
     print(item)
  print("*"* 100)


  # print(positive_FR.values()[:10])
  # print(neutral_FR.values()[:10])
  # print(negative_FR.values()[:10])
  with open("Refine/Positive_FR_" + str(template_id) + "_one.txt", "w") as f:
    for word in positive_FR:
      f.write(word + "\n")

  with open("Refine/Neutral_FR_" + str(template_id) + "_one.txt", "w") as f:
    for word in neutral_FR:
      f.write(word + "\n")

  with open("Refine/Negative_FR_" + str(template_id) + "_one.txt", "w") as f:
    for word in negative_FR:
      f.write(word + "\n")

  print(template_id, "*"*100)