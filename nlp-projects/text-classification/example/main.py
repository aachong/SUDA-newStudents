import dataLoader
import avgModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import LabelSmoothing

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = dataLoader.train_data
test_data = dataLoader.test_data
vocab_size = dataLoader.MAX_WORD_SIZE
index2label = dataLoader.index2label
index2word = dataLoader.index2word
embedding_size = 128
pad_idx = 0
output_size = 5

model = avgModule.avgModule(
    vocab_size, embedding_size, pad_idx, output_size).to(device)

# for p in model.parameters():
#         if p.dim() > 1:
#             nn.init.xavier_uniform(p)
criterion = nn.CrossEntropyLoss().to(device)
# criterion = LabelSmoothing().to(device)
optimizer = optim.Adam(model.parameters())
train_data[0][2]


def accuracy(preds, y):
    f_preds = preds.max(1)[1]
    correct = (f_preds == y).float()
    acc = sum(correct)/len(correct)
    return acc


def evaluate(model, criterion, data):
    epoch_acc = 0.0
    epoch_loss = 0.0
    for (x, length, y) in data:
        x = torch.from_numpy(x).long().to(device)
        y = torch.from_numpy(y).long().to(device)
        preds = model(x)

        loss = criterion(preds, y)
        acc = accuracy(preds, y)
        epoch_acc += acc
        epoch_loss += loss
    print(f'测试集：准确率:{epoch_acc/len(data)},loss:{epoch_loss/len(data)}')
    return epoch_acc/len(data)


def train(model, criterion, optimizer, data):
    epochs = 100
    maxv = 0.4
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_acc = 0
        for (x, length, y) in data:
            x = torch.from_numpy(x).long().to(device)
            y = torch.from_numpy(y).long().to(device)
            preds = model(x)

            loss = criterion(preds, y)
            acc = accuracy(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            epoch_acc += acc
        with torch.no_grad():
            model.eval()
            tmp = evaluate(model, criterion, test_data)
            model.train()
            if(tmp > maxv):
                maxv = tmp
                torch.save(model.state_dict(), 'avgModel.pt')
        print(
            f'Epoch:{epoch},精准度:{epoch_acc/len(data)},loss:{epoch_loss/len(data)}')


train(model, criterion, optimizer, train_data)

model.load_state_dict(torch.load('avgModel.pt'))

evaluate(model, criterion, test_data)

it = iter(model.parameters())
para = next(it)
