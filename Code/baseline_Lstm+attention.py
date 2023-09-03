
import A_load_data
from konlpy.tag import Hannanum
import fasttext
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score
import random
import math

import argparse 
parser = argparse.ArgumentParser(description="models setting")
parser.add_argument('--learning_rate', dest='learning_rate', type=float, help='learning_rate', default=5e-5)
parser.add_argument('--epoch', dest='epoch', type=int, help='epoch', default=100)
parser.add_argument('--iter', dest='iter', type=int, help='iter', default=0)

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
epoches = args.epoch

batch_size = 64
hidden_size = 300
n_class = 3
maxlen = 100
randomseed = 3407

torch.cuda.manual_seed(randomseed + args.iter)
torch.manual_seed(randomseed + args.iter)
np.random.seed(randomseed + args.iter)
random.seed(randomseed + args.iter)


hannanum = Hannanum()
ft = fasttext.load_model("Fasttext/cc.ko.300.bin")

X_train, Y_train = A_load_data.load_weibo_train()
X_test, Y_test = A_load_data.load_weibo_test()
sentence_vectors_train = []
for sentence in X_train:
    words = hannanum.morphs(sentence)
    sent_len = len(words)
    if sent_len > 100:
        words = words[:100]
    else:
        for i in range(100-sent_len):
            words.append("<PAD>")
    vectors = []
    for word in words:
        id = ft.get_word_vector(word)
        vectors.append(id)
    vectors = np.vstack(vectors)
    sentence_vectors_train.append(vectors)
sentence_vectors_test = []
for sentence in X_test:
    words = hannanum.morphs(sentence)
    sent_len = len(words)
    if sent_len > 100:
        words = words[:100]
    else:
        for i in range(100-sent_len):
            words.append("<PAD>")
    vectors = []
    for word in words:
        id = ft.get_word_vector(word)
        vectors.append(id)
    vectors = np.vstack(vectors)
    sentence_vectors_test.append(vectors)


class CNNDataset(Data.Dataset):
  def __init__(self, sentences, labels):
    self.sentences = sentences
    self.labels = labels
  def __len__(self):
    return len(self.sentences)

  def __getitem__(self, index):
    sent = self.sentences[index]
    sent = torch.from_numpy(sent).float().unsqueeze(0)
    label = self.labels[index]
    return sent, label
        
class MyLSTM(nn.Module):
    def __init__(self):
        super(MyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=300, hidden_size=300, num_layers=2, bidirectional=True, dropout=0.5, batch_first=True)
        self.linear = nn.Linear(in_features=300*2, out_features=3)
        self.dropout = nn.Dropout(0.5)

    def attention_net(self,lstm_output, final_state):
        # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
        # final_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        # hidden = final_state.view(batch_size,-1,1)
        hidden = torch.cat((final_state[0],final_state[1]),dim=1).unsqueeze(2)
        # hidden : [batch_size, n_hidden * num_directions(=2), n_layer(=1)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        # attn_weights : [batch_size,n_step]
        soft_attn_weights = F.softmax(attn_weights,1)

        # context: [batch_size, n_hidden * num_directions(=2)]
        context = torch.bmm(lstm_output.transpose(1,2),soft_attn_weights.unsqueeze(2)).squeeze(2)

        return context, soft_attn_weights
    
    def forward(self, inputs):
        inputs = inputs.squeeze(1)
        output, hidden_tuple = self.lstm(inputs)
        hidden = hidden_tuple[0]
        
        hidden = self.dropout(hidden)

        attn_output, alpha_n = self.attention_net(output, hidden)
        
        attn_output = self.dropout(attn_output)
        return self.linear(attn_output) 

encoder = OneHotEncoder(categories="auto", sparse=False)
Y_train = encoder.fit_transform(np.expand_dims(np.array(Y_train), axis=-1))
Y_train = torch.from_numpy(Y_train)
Y_test = encoder.fit_transform(np.expand_dims(np.array(Y_test), axis=-1))
Y_test = torch.from_numpy(Y_test)
train_loader = Data.DataLoader(dataset= CNNDataset(sentence_vectors_train, Y_train), batch_size=batch_size, shuffle=True,)
test_loader = Data.DataLoader(dataset= CNNDataset(sentence_vectors_test, Y_test), batch_size=batch_size, shuffle=False)

rnn = MyLSTM().to(device)

optimizer = optim.Adam(rnn.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()   

rnn.train()
for epoch in range(epoches):
    loss_list = []
    for x_t, y_t in train_loader:
        optimizer.zero_grad()
        pred = rnn(x_t.to(device))
        loss = F.cross_entropy(pred, y_t.argmax(dim=-1).to(device), reduction="mean")
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
    if epoch%10 ==0 or epoch == epoches-1:
        print('[{}|{}]  loss:{:.4f}'.format(epoch+1, epoches, np.mean(np.array(loss_list))))

rnn.eval()
true_list, preds_list = [], []
for batch in test_loader:
    with torch.no_grad():
        batch = tuple(p.to(device) for p in batch)
        true_list.append(batch[1].detach().cpu().numpy())
        preds = rnn(batch[0])
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
