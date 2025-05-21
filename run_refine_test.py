import json
import os

from models import get_model
from modules.actions import *

with open('./data/criteria/evaluation_metrics.json', 'r', encoding = 'utf-8') as file:
    eval_metrics = json.load(file)
with open('./data/criteria/metrics_map.json', 'r', encoding = 'utf-8') as file:
    metrics_map = json.load(file)

criterias = {
    theme: [
        eval_metrics[int(metric[0]) - 1]['sub_metrics'][int(metric[2]) - 1]
        for metric in metrics
    ]
    for theme, metrics in metrics_map.items()
}

raw_datas = []
data_dir = './data_raw/example_data_20250425/example_zh_only_20250425/filtered_zh_data_sampled_annotation_without_model_name'
for path in os.listdir(data_dir):
    with open(os.path.join(data_dir, path), 'r', encoding = 'utf-8') as file:
        json_obj = json.load(file)
        raw_datas.append({
            'original_data': {
                'message': [
                    {'role': 'user', 'content': json_obj['question']},
                    {'role': 'assistant', 'content': json_obj['raw_answer']}
                ],
                'theme': json_obj['question_type_ZH']
            }
        })

models = ['qwen2.5-7b-instruct', 'qwen2.5-14b-instruct', 'qwen-max', 'deepseek-v3', 'deepseek-r1']
models = [get_model(model) for model in models]

for data in raw_datas:
    state = data['original_data']
    state['criteria'] = criterias[state['theme']]
    for model in models:
        eval_node = Evaluate()
        state = eval_node(state = state, llm = model)
    
    refine_node = Refine()
    state = refine_node(state = state, llm = models[-1])
        
    for model in models:
        eval_node = Evaluate()
        state = eval_node(state = state, llm = model)
    
    print(state)