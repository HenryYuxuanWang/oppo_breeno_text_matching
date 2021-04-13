import os
import json

import numpy as np
from utils import truncate_sequences

cache_dir = '/data/pretrained_models/torch/bert'
pretrained_model = 'bert-base-chinese'
dict_path = os.path.join(cache_dir, *[pretrained_model, 'vocab.txt'])

min_count = 5
maxlen = 32


def load_data(filename):
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
            truncate_sequences(maxlen, -1, a, b)
            D.append((a, b, c))
    return D


# 加载数据集
data = load_data('/data/oppo_breeno/oppo_breeno_round1_data/gaiic_track3_round1_train_20210228.tsv')
train_data = [d for i, d in enumerate(data) if i % 10 != 0]
valid_data = [d for i, d in enumerate(data) if i % 10 == 0]
test_data = load_data('/data/oppo_breeno/oppo_breeno_round1_data/gaiic_track3_round1_testA_20210228.tsv')

# 统计词频
tokens = {}
for d in data + test_data:
    for i in d[0] + d[1]:
        tokens[i] = tokens.get(i, 0) + 1

tokens = {i: j for i, j in tokens.items() if j >= min_count}
tokens = sorted(tokens.items(), key=lambda s: -s[1])
tokens = {t[0]: i + 7 for i, t in enumerate(tokens)}  # 0: pad, 1: unk, 2: cls, 3: sep, 4: mask, 5: no, 6: yes

# BERT词频
counts = json.load(open('counts.json'))
del counts['[CLS]']
del counts['[SEP]']
token_dict = {}
with open(dict_path, encoding='utf-8') as reader:
    for line in reader:
        token = line.split()
        token = token[0] if token else line.strip()
        token_dict[token] = len(token_dict)
freqs = [counts.get(i, 0) for i, j in sorted(token_dict.items(), key=lambda s: s[1])]  # bert vocab中每个token的词频
keep_tokens = list(np.argsort(freqs)[::-1])  # 最高频token的index列表

keep_tokens = [0, 100, 101, 102, 103, 100, 100] + keep_tokens[:len(tokens)]
keep_tokens_dict = {i: num for i, num in enumerate(keep_tokens)}

# 模拟未标注
for d in valid_data + test_data:
    train_data.append((d[0], d[1], -5))

if __name__ == '__main__':
    from pytorch_model import Processor, Model
    from training_args import args

    processor = Processor(tokens, keep_tokens_dict)

    train_data = processor.get_examples(train_data, random=True)
    valid_data = processor.get_examples(valid_data)

    model = Model()
    model.train(train_data, valid_data, args=args, keep_tokens=keep_tokens)
    # model.train(train_data, valid_data, args=args, checkpoint='/mnt/data/yuxuan/match/oppo_breeno/results')
