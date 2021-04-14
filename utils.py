import numpy as np


def truncate_sequences(max_len, index, *sequences):
    """截断总长度至不超过max_len
    """
    sequences = [s for s in sequences if s]
    while True:
        lengths = [len(s) for s in sequences]
        if sum(lengths) > max_len:
            i = np.argmax(lengths)
            sequences[i].pop(index)
        else:
            return sequences


def load_data(filename, max_len):
    """加载数据
    单条格式：(文本1 ids, 文本2 ids, 标签id)
    """
    D = []
    with open(filename) as f:
        for l in f:
            l = l.strip().split('\t')
            if len(l) == 3:
                a, b, c = l[0], l[1], int(l[2])
            else:
                a, b, c = l[0], l[1], -5  # 未标注数据，标签为-5
            a = [int(i) for i in a.split(' ')]
            b = [int(i) for i in b.split(' ')]
            truncate_sequences(max_len, -1, a, b)
            D.append((a, b, c))
    return D


def load_vocab(dict_path, encoding='utf-8'):
    """从bert的词典文件中读取词典
    """
    token_dict = {}
    with open(dict_path, encoding=encoding) as reader:
        for line in reader:
            token = line.split()
            token = token[0] if token else line.strip()
            token_dict[token] = len(token_dict)
    return token_dict
