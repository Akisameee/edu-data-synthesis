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