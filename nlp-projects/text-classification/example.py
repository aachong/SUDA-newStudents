from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

MAX_WORD_SIZE = 60000

def load_data(filename): 
    x,y = [],[]
    with open(filename,'r',encoding='utf-8') as f:
        for line in f :
            line = line.strip().split('\t')
            x.append(line[1])
            y.append(line[0])
    return x,y

train_input,train_output = load_data('train.txt')
test_input,test_output = load_data('test.txt')

print(f'输入:{train_input[3][:20]},输出:{train_output[3]}')
print(f'输入:{test_input[3][:20]},输出:{test_output[3]}')

def build_vocab(text):
    all_c = ' '.join(text)
    all_c = all_c.split(' ')
    dic = Counter(all_c)
    dic = dic.most_common(MAX_WORD_SIZE-1)
    dic,_ = zip(*dic)
    index2word = ['<PAD>']+list(dic)
    word2index = {word:index for index,word in enumerate(index2word)}
    return index2word,word2index

index2word,word2index = build_vocab(train_input)
for i in train_input:
    a = 0
    a = max(a,len(i))
print(a)




dic = Counter()
for i in train_input:
    words = i.split(' ')
    for w in words:
        dic[w]+=1

ls = [i.split(' ') for i in train_input]

d = Counter(ls)
d.most_common(10)
len(dic)

