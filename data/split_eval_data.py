import random
import sys
sys.path.insert(0, '.')

from modules.utils import *

TRAIN_SIZE = 0.6

eval_datas = read_jsonl('./data/eval_data/eval_samples.jsonl')
eval_datas = [d for d in eval_datas if d['eval'].startswith('human_')]
print(len(eval_datas))
id_to_datas = {}
for data in eval_datas:
    if data['id'] not in id_to_datas:
        id_to_datas[data['id']] = []
    id_to_datas[data['id']].append(data)

train_datas = []
val_datas = []
for language in ['zh', 'en']:
    for qid in range(99):
        mids = list(range(5))
        random.shuffle(mids)
        split_idx = int(TRAIN_SIZE * len(mids))
        for mid in mids[:split_idx]:
            train_datas += id_to_datas[
                f'{language}_question_{qid}_model_{mid}.json'
            ]
        for mid in mids[split_idx:]:
            val_datas += id_to_datas[
                f'{language}_question_{qid}_model_{mid}.json'
            ]

print(f'train set size: {len(train_datas)}')
print(f'val set size: {len(val_datas)}')
write_jsonl('./data/eval_data/train_eval_data.jsonl', train_datas)
write_jsonl('./data/eval_data/val_eval_data.jsonl', val_datas)