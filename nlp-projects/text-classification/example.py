from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def load_data(filename): 
    x,y = [],[]
    with open(filename,'r',encoding='utf-8') as f:
        for line in f :
            line = line.strip().split('\t')
            x.append(line[1])
            y.append(line[0])
    return x,y

train_input,train_output = load_data('train.txt')
print(train_input[:3])
print(train_output[:3])

def encoding

import jieba