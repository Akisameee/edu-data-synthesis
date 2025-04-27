import os
import json

import sys
sys.path.insert(0, '.')

from data.utils import *

type_map = {
    'MCQs': 'single_choice',
    'Fill_in_Blanks': 'fill_in_blank',
    '多选题': 'multiple_choice',
    '填空题': 'fill_in_the_blank',
    '判断题': 'true_or_false',
    'Open-ended_Questions': 'short_answer'
}

def process_gaokao_json(datas: dict):

    subject = datas['keywords'].split('_')[1]
    type_ = '_'.join(datas['keywords'].split('_')[1:]).lower()
    processed_datas = []
    for data in datas['example']:
        data.pop('index')
        data['subject'] = subject.lower()
        data['type'] = type_
        data['level'] = 'senior'
        data['source'] = 'gaokao-bench'

        processed_datas.append(data)

    return processed_datas

gaokao_datas = []
root_dir = './data_raw/gaokao-bench'

for path in yield_json_files(root_dir):
    with open(path, 'r', encoding = 'utf-8') as file:
        json_obj = json.load(file)
    
    gaokao_datas += process_gaokao_json(json_obj)

save_path = './data/zh/gaokao-bench.jsonl'
with open(save_path, 'w', encoding = 'utf-8') as file:
    for gaokao_data in gaokao_datas:
        file.write(json.dumps(gaokao_data, ensure_ascii = False) + '\n')
