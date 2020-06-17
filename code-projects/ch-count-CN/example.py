def read_file(filename):
    with open('textCN.txt', 'r', encoding='utf-8') as f:
        context = [line for line in f]
    return context


lines = read_file('33')
print(lines[:3])
lines = ' '.join(lines)
dic = {}
f = 1
for word in lines:
    if word not in dic:
        dic[word] = 0
    else:
        dic[word] += 1
for i in dic.items():
    print(i)
dicc = sorted(dic.items(), key=lambda x: x[1], reverse=True)
dicc[:5]
