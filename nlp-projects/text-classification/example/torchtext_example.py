import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data


class raw_data:
    def __init__(self, typee):
        self.input,self.output = self.load_data('../'+ typee +'.txt')

    def load_data(self, filename):
        x, y = [], []
        with open(filename, 'r', encoding='utf-8') as f:
            for i in f.readlines():
                lines = i.strip().split('\t')
                x.append(lines[1])
                y.append(lines[0])
        return x, y

def getExample(input:raw_data, TEXT, LABEL):
    example = []
    field = [('text', TEXT), ('label', LABEL)]
    for t, l in zip(input.input, input.output):
        example.append(data.Example.fromlist([t, l], field))
    return example, field

def get():
    # 读取数据
    train_Rdata = raw_data('train')
    test_Rdata = raw_data('test')
    print(test_Rdata.input[0])
    # 创建field
    TEXT = data.Field(tokenize=lambda x: x.split(' '),batch_first=True,fix_length=200)
    LABEL = data.Field(batch_first=True,sequential=False,is_target=True,unk_token=None,pad_token=None)
    # 构造example
    train_example, train_field = getExample(train_Rdata,TEXT, LABEL)
    test_example, test_field = getExample(test_Rdata,TEXT, LABEL)
    # 构造dataset
    train_dataset = data.Dataset(train_example, train_field)
    test_dataset = data.Dataset(test_example, test_field)
    # 传回给field
    TEXT.build_vocab(train_dataset,max_size=60000-2)
    LABEL.build_vocab(train_dataset)
    # 构造iter
    # x_iter,t_iter = data.BucketIterator.splits((train_dataset,test_dataset), (64,64),device='cuda',sort_key=lambda x: x.text)
    # len(TEXT.vocab.itos)
    return data.BucketIterator.splits((train_dataset,test_dataset), (64,64),device='cuda',sort_key=lambda x: x.text)
    # next(iter(t_iter)).text


