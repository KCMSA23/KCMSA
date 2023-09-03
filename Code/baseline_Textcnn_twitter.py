
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random

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

X_train, Y_train = A_load_data.load_twitter_train()
X_test, Y_test = A_load_data.load_twitter_test()
sentence_vectors_train = []
for sentence in X_train:
    words = hannanum.morphs(sentence)
    sent_len = len(words)
    if sent_len > 100:
        words = words[:100]
    else:
        for i in range(100-sent_len):
            words.append("<UNK>")
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
            words.append("<UNK>")
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

class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.conv3 = nn.Conv2d(1, 1, (3, hidden_size))
        self.conv4 = nn.Conv2d(1, 1, (4, hidden_size))
        self.conv5 = nn.Conv2d(1, 1, (5, hidden_size))
        self.Max3_pool = nn.MaxPool2d((100-3+1, 1))
        self.Max4_pool = nn.MaxPool2d((100-4+1, 1))
        self.Max5_pool = nn.MaxPool2d((100-5+1, 1))
        self.linear1 = nn.Linear(3, n_class)

    def forward(self, x):
        batch = x.shape[0]
        # Convolution
        x1 = F.relu(self.conv3(x))
        x2 = F.relu(self.conv4(x))
        x3 = F.relu(self.conv5(x))

        # Pooling
        x1 = self.Max3_pool(x1)
        x2 = self.Max4_pool(x2)
        x3 = self.Max5_pool(x3)
        # capture and concatenate the features
        x = torch.cat((x1, x2, x3), -1)
        x = x.view(batch, -1)

        # project the features to the labels
        x = self.linear1(x)
        return x



encoder = OneHotEncoder(categories="auto", sparse=False)
Y_train = encoder.fit_transform(np.expand_dims(np.array(Y_train), axis=-1))
Y_train = torch.from_numpy(Y_train)
Y_test = encoder.fit_transform(np.expand_dims(np.array(Y_test), axis=-1))
Y_test = torch.from_numpy(Y_test)
train_loader = Data.DataLoader(dataset= CNNDataset(sentence_vectors_train, Y_train), batch_size=batch_size, shuffle=True,)
test_loader = Data.DataLoader(dataset= CNNDataset(sentence_vectors_test, Y_test), batch_size=batch_size, shuffle=False)

cnn = TextCNN().to(device)
optimizer = optim.Adam(cnn.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()   

cnn.train()
for epoch in range(epoches):
    loss_list = []
    for x_t, y_t in train_loader:
        optimizer.zero_grad()
        
        pred = cnn(x_t.to(device))
        loss = F.cross_entropy(pred, y_t.argmax(dim=-1).to(device), reduction="mean")
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
    if epoch%10 ==0 or epoch == epoches-1:
        print('[{}|{}]  loss:{:.4f}'.format(epoch+1, epoches, np.mean(np.array(loss_list))))

cnn.eval()
true_list, preds_list = [], []
for batch in test_loader:
    with torch.no_grad():
        batch = tuple(p.to(device) for p in batch)
        true_list.append(batch[1].detach().cpu().numpy())
        preds = cnn(batch[0])
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
