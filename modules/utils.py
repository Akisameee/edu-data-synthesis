import os
import json
import re
import inspect
from typing import get_type_hints, Optional, List, Tuple, Any
import functools
from tqdm import tqdm

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
            try:
                json_obj = fix_json_close(json_str)
                return json_obj
            except:
                raise ValueError(f'[JSON Parse Error] {str(e)}.')
    else:
        return extract_json(f'```json{response}```')
        # raise ValueError(f'[JSON Parse Error] Code block not found. Invalid response: {response}')

def fix_json_close(json_str: str):
    close_prefix, close_suffix = [], []
    for idx in range(len(json_str)):
        if json_str[idx] in ['[', '{']:
            close_prefix.append(json_str[idx])
        else: break
    for idx in range(len(json_str) - 1, 0, -1):
        if json_str[idx] in [']', '}']:
            close_suffix.append(json_str[idx])
        else: break
    
    if len(close_prefix) > len(close_suffix):
        json_str = json_str[len(close_prefix) - len(close_suffix):]
    elif len(close_prefix) < len(close_suffix):
        json_str = json_str[: -(len(close_suffix) - len(close_prefix))]
    return json.loads(json_str)

def extract_boxed(response: str):
    pattern = r"\\boxed{(.*)}"
    match = re.search(pattern, response)
    if match:
        boxed_str = match.group(1).strip()
        if boxed_str[0].isupper():
            return boxed_str[0]
        else:
            raise ValueError(f'[Boxed Parse Error] Invalid boxed string: {boxed_str}')
    else:
        ValueError(f'[Boxed Parse Error] \\boxed not found. Invalid response: {response}')

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

def retry(
    max_attempt: int = 3,
    verbose: bool = False
):
    def decorator(func):
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                n_attempt = 0
                while n_attempt < max_attempt:
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        n_attempt += 1
                        if verbose:
                            tqdm.write(f"Attempt {n_attempt}/{max_attempt} failed: {e}")
                        if n_attempt == max_attempt:
                            raise e
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                n_attempt = 0
                while n_attempt < max_attempt:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        n_attempt += 1
                        if verbose:
                            tqdm.write(f"Attempt {n_attempt}/{max_attempt} failed: {e}")
                        if n_attempt == max_attempt:
                            raise e
            return sync_wrapper
    return decorator