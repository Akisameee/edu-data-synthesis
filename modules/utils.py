import os
import json
import re
import inspect
from typing import get_type_hints, Optional, List, Tuple, Any

def yield_json_files(root_dir: str):

    for walk_res in os.walk(root_dir):
        for filename in walk_res[2]:
            file_path = os.path.join(walk_res[0], filename)
            if file_path.endswith('.json') or file_path.endswith('.jsonl'):
                yield file_path

def extract_json(response: str):

    match = re.search(r'```json\s*(.*)\s*```', response, re.DOTALL)

    if match:
        json_str = match.group(1)

        try:
            json_obj = json.loads(json_str)
            return json_obj 
        except Exception as e:
            raise ValueError(f'[JSON Parse Error] {str(e)}. Invalid JSON string: {json_str}')
    else:
        raise ValueError(f'[JSON Parse Error] Code block not found. Invalid response: {response}')
    
def read_criterias(metrics_dir: str):

    with open(os.path.join(metrics_dir, 'evaluation_metrics_old.json'), 'r', encoding = 'utf-8') as file:
        eval_metrics = json.load(file)
    with open(os.path.join(metrics_dir, 'metrics_map.json'), 'r', encoding = 'utf-8') as file:
        metrics_map = json.load(file)

    criterias = {
        theme: [
            eval_metrics[int(metric[0]) - 1]['sub_metrics'][int(metric[2]) - 1]
            for metric in metrics
        ]
        for theme, metrics in metrics_map.items()
    }

    return criterias

def read_scenarios(theme_dir: str, language: str):

    with open(os.path.join(theme_dir, f'{language}_scenario.json'), 'r', encoding = 'utf-8') as file:
        scenarios = json.load(file)

    return scenarios

def inspect_method(cls, method_name: str) -> List[Tuple[str, Optional[type]]]:

    methods = inspect.getmembers(cls, predicate=inspect.isfunction)
    
    params = []
    for name, method in methods:
        if name == method_name:
            unwarpped_method = inspect.unwrap(method)
            signature = inspect.signature(unwarpped_method)
            parameters = signature.parameters
            type_hints = get_type_hints(unwarpped_method)
            for param_name, param in parameters.items():
                param_type = type_hints.get(param_name, None)
                params.append((param_name, param_type))

    return params

def read_jsonl(path: str):

    json_objs = []
    with open(path, 'r', encoding = 'utf-8') as file:
        for idx, line in enumerate(file.readlines()):
            try:
                json_obj = json.loads(line)
                json_objs.append(json_obj)
            except Exception as e:
                print(f'Line: {idx}, Error: {e}')

    return json_objs

def write_jsonl(path: str, json_objs: list):

    with open(path, 'w', encoding = 'utf-8') as file:
        for json_obj in json_objs:
            file.write(json.dumps(json_obj, ensure_ascii = False) + '\n')

def read_sampled_data(language: str):

    datas = []
    zh_dir = f'./data_raw/{language}_data_sampled/'
    for path in os.listdir(zh_dir):
        with open(os.path.join(zh_dir, path), 'r', encoding = 'utf-8') as file:
            data = json.load(file)
            data['language'] = language
            datas.append(data)

    return datas