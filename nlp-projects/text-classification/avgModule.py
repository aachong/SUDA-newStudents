import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class avgModule(nn.Module):
    def __init__(self, vocab_size, embedding_size, pad_idx, output_size):
        super(avgModule, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, pad_idx)
        self.fc = nn.Linear(embedding_size, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input):
        # input : batch_size,vocab_size

        embeded = self.embedding(input)  # b_size,vocab_size,embeding_size
        embeded = self.dropout(embeded)
        p1 = F.avg_pool2d(embeded, (len(embeded[1]), 1)).squeeze()
        p2 = F.max_pool2d(embeded, (len(embeded[1]), 1)).squeeze()
        # pooled=p1+p2
        pooled=p2
        return self.fc(pooled)