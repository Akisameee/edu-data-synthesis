import os
import json

import sys
sys.path.insert(0, '.')

from data.utils import *

subject_map = {
    '初中信息技术': 'computer_science',
    '初中化学': 'chemistry',
    '初中历史': 'history',
    '初中地理': 'geography',
    '初中数学': 'math',
    '初中物理': 'physics',
    '初中生物': 'biology',
    '初中科学': 'science',
    '初中英语': 'english',
    '初中语文': 'chinese',
}

type_map = {
    '选择题': 'single_choice',
    '单选题': 'single_choice',
    '多选题': 'multiple_choice',
    '填空题': 'fill_in_blank',
    '判断题': 'true_or_false',
    '简答题': 'short_answer'
}

def process_cjeval_json(datas: list):

    processed_datas = []
    for data in datas:
        data['subject'] = subject_map[data['subject']]

        # data['type'] = type_map.get(data['ques_type'], 'short_answer')
        data['type'] = data['ques_type']
        del data['ques_type']

        data['difficulty'] = data['ques_difficulty']
        del data['ques_difficulty']

        data['question'] = data['ques_content']
        del data['ques_content']

        data['answer'] = data['ques_answer']
        del data['ques_answer']

        data['analysis'] = data['ques_analyze']
        del data['ques_analyze']

        data['knowledges'] = data['ques_knowledges']
        del data['ques_knowledges']

        data['level'] = 'junior'
        data['source'] = 'cjeval'

        processed_datas.append(data)

    return processed_datas

cjeval_datas = []
root_dir = './data_raw/cjeval'

for path in yield_json_files(root_dir):
    json_obj = []
    with open(path, 'r', encoding = 'utf-8') as file:
        for line in file.readlines():
            json_obj.append(json.loads(line))
    
    cjeval_datas += process_cjeval_json(json_obj)

save_path = './data/zh/cjeval.jsonl'
with open(save_path, 'w', encoding = 'utf-8') as file:
    for cjeval_data in cjeval_datas:
        file.write(json.dumps(cjeval_data, ensure_ascii = False) + '\n')
