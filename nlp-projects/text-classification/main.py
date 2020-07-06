import dataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

train_data = dataLoader.train_data
test_data = dataLoader.test_data
vocab_size = dataLoader.MAX_WORD_SIZE


class module(nn.Module):
    def __init__(self,vocab_size,embedding_size):
        super(module,self).__init__()
        self.embedding = nn.Embedding(vocab_size,128)
        self.fc1 = nn.Linear(128,64)
        self.fc2 = nn.Linear(64,5)

    def forward(self,input):
        #batch_size,sequence_len
        embeded = self.embedding(input)
        fc1 = self.fc1(embeded)
        pooled = F.avg_pool2d(fc1,(len(fc1[1]),1)).squeeze()
        

