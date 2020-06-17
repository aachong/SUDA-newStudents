def load_data(file):
    with open(file, 'r', encoding='utf-8') as f:
        return [lines.strip() for lines in f]


sentences = load_data('textCN.txt')
targets = load_data('textCN.gold')
dicts = load_data('CN.dict')
dicts = {w: 0 for w in dicts[1:]}
sentences[-4:]
targets[-4:]
dicts


def participle(sentences):
    words = []
    text = ' '.join(sentences)
    text_size = len(text)
    c = 0
    while c < text_size:
        if text[c] == ' ':
            c += 1
        for i in range(10, 0, -1):
            if c+i > text_size:
                continue
            if text[c:c+i] in dicts:
                words.append(text[c:c+i])
                c = c+i-1
                break
            if i == 1:
                words.append(text[c:c+i])
                c = c+i-1
        c += 1
    return words


par_sentence = participle(sentences)
results = ' '.join(par_sentence)
targets = ' '.join(targets)
print(targets[:100])
print(results[:100])


def lens(s):
    count = 0
    for c in s:
        if c != ' ':
            count += 1
    return count


print(lens(results))


def evaluate(targets, results):
    right, number_t, number_r = 0, 0, 0
    targets = targets.split(' ')
    results = results.split(' ')
    number_t = len(targets)
    number_r = len(results)
    diff = 0
    it_r = iter(results)
    it_t = iter(targets)
    while True:
        if diff == 0:
            word_r = next(it_r, None)
            word_t = next(it_t, None)
            if word_r == None:
                break
            if len(word_t) == len(word_r):
                right += 1
                continue
            else:
                diff = len(word_t)-len(word_r)
        if diff > 0:
            while diff > 0:
                word_r = next(it_r)
                diff -= len(word_r)
                print(word_r,word_t,sep=' ')
        if diff < 0:
            while diff < 0:
                word_t = next(it_t)
                diff += len(word_t)
                print(word_r,word_t,sep=' ')
    return right,number_t,number_r

right, number_t, number_r = evaluate(targets,results)
print(right,number_t,number_r,sep=' ')
precision = right/number_r
recall = right/number_t
F = precision*recall*2/(precision+recall)

print('召回率',precision,sep='=')
print('准确率',recall,sep='=')
print('F值',F,sep='=')

