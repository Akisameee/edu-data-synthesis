import os
import json
import re

def yield_json_files(root_dir):

    for walk_res in os.walk(root_dir):
        for filename in walk_res[2]:
            file_path = os.path.join(walk_res[0], filename)
            if file_path.endswith('.json') or file_path.endswith('.jsonl'):
                yield file_path

def extract_json(response: str):

    match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)

    if match:
        json_str = match.group(1)

        try:
            json_obj = json.loads(json_str)
            return json_obj 
        except Exception as e:
            raise ValueError(f'[JSON Parse Error] {str(e)}. Invalid JSON string: {response}')
    else:
        raise ValueError(f'[JSON Parse Error] Code block not found. Invalid json string: {response}')
    
def read_criterias(metrics_path: str, map_path: str):

    with open(metrics_path, 'r', encoding = 'utf-8') as file:
        eval_metrics = json.load(file)
    with open(map_path, 'r', encoding = 'utf-8') as file:
        metrics_map = json.load(file)

    criterias = {
        theme: [
            eval_metrics[int(metric[0]) - 1]['sub_metrics'][int(metric[2]) - 1]
            for metric in metrics
        ]
        for theme, metrics in metrics_map.items()
    }

    return criterias