from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


MAX_WORD_SIZE = 60000
BATCH_SIZE = 64


def load_data(filename):
    x, y = [], []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            x.append(line[1])
            y.append(line[0])
    return x, y


train_input, train_output = load_data('train.txt')
test_input, test_output = load_data('test.txt')

print(f'输入:{train_input[3][:20]},输出:{train_output[3]}')
print(f'输入:{test_input[3][:20]},输出:{test_output[3]}')


def build_vocab(text):
    all_c = ' '.join(text)
    all_c = all_c.split(' ')
    dic = Counter(all_c)
    dic = dic.most_common(MAX_WORD_SIZE-2)
    dic, _ = zip(*dic)
    index2word = ['<PAD>']+['<UNK>']+list(dic)
    word2index = {word: index for index, word in enumerate(index2word)}
    return index2word, word2index


index2word, word2index = build_vocab(train_input)
count_out = Counter(train_output)
index2label = list(count_out)
label2index = {w: i for i, w in enumerate(index2label)}


def encode(train_input, train_output, word2index, label2index):

    train_input1 = [[word2index.get(w, 1)
                     for w in i.split(' ')] for i in train_input]
    train_output1 = [[label2index.get(w, 1)
                      for w in i.split(' ')] for i in train_output]

    sorted_index = sorted(range(len(train_input1)),
                          key=lambda x: len(train_input1[x]))
    out_input = [train_input1[i] for i in sorted_index]
    out_output = [train_output1[i] for i in sorted_index]
    return out_input, out_output


train_in, train_out = encode(
    train_input, train_output, word2index, label2index)
print(' '.join([index2word[i] for i in train_in[1000]]))
print(' '.join([index2label[i] for i in train_out[1000]]))


def get_batch_index(n, batch_index_size):
    idx_list = np.arange(0, n, batch_index_size)
    np.random.shuffle(idx_list)
    batch_index = []
    for i in idx_list:
        batch_index.append(np.arange(i, min(batch_index_size+i, n)).tolist())
    return batch_index


def process_data(sentences, label):
    lens = [len(s) for s in sentences]
    n = len(sentences)
    max_len = np.max(lens)

    out = np.zeros((n, max_len)).astype('int32')
    for i in range(n):
        out[i, :lens[i]] = sentences[i]
    return out, np.array(lens).astype('int32'), np.array(label).astype('int32')


def get_batch(train_in, train_out, batch_size):
    batches_index = get_batch_index(len(train_in), batch_size)
    batches_data = []
    for batch in batches_index:
        sentences = [train_in[i]for i in batch]
        labal = [train_out[i][0] for i in batch]
        batches_data.append(process_data(sentences, labal))
    return batches_data


# [(train_in[i],train_out[i]) for i in range(3)]
train_data = get_batch(train_in, train_out, BATCH_SIZE)
# batch，句子，句子长度，标签


class avgModule(nn.Module):
    def __init__(avgModule,self,vocab_size,embedding_size,pad_idx,output_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_size,pad_idx)
        self.fc = nn.Linear(embedding_size,output_size)

    def forward(self,input):
        # input : batch_size,1,vocab_size
        input.squeeze_()
        embeded = self.embedding(input) #b_size,vocab_size,embeding_size
        avged = F.avg_pool2d(embeded,(embeded[1],1))


a = torch.randn(5,4,3)
b = F.avg_pool2d(a,(a.shape[1],1))
b.shape