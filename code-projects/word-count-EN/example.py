from collections import Counter
import nltk

def load_data():
    en = []
    with open('textEN.txt', 'r', encoding='utf-8') as f:
        sentence = [line.strip() for line in f]
        word = [nltk.word_tokenize(s) for s in sentence]
        return word

en_word = load_data()
en_word

def build_dict(sentences,max_size):
    dic = Counter()
    for sentence in sentences:
        for word in sentence:
            dic[word]+=1
    ls = dic.most_common(max_size-2)
    index2word = ['unk','pad']+[w[0] for w in ls]
    word2index = {word[0]:index+2 for index,word in enumerate(ls)}
    return ls,index2word,word2index

ls,index2word,word2index = build_dict(en_word,5000)
index2word[:10]
word2index['and']


