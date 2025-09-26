import os
import json
import yaml
import inspect
import hashlib
import aiofiles
from typing import get_type_hints, Optional, Dict, List, Tuple, Any

def load_config(config_path: str = './config.yaml') -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file does not exist: {config_path}')
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config or {}
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f'Config parse error: {e}')

def get_config_value(key_path: str, config_path: str = './config.yaml', default: Any = None) -> Any:
    keys = key_path.split('.')
    current = load_config(config_path)
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current

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
            except json.JSONDecodeError as e:
                print(f'Line: {idx}, Error: {e}')
            except Exception as e:
                print(f'Line: {idx}, Unexpected Error: {e}')
    return json_objs

def write_jsonl(path: str, json_objs: list, append: bool = False):
    mode = 'a' if append else 'w'
    with open(path, mode, encoding = 'utf-8') as file:
        for json_obj in json_objs:
            file.write(json.dumps(json_obj, ensure_ascii = False) + '\n')

async def aread_jsonl(path: str) -> List[Any]:
    json_objs = []
    async with aiofiles.open(path, 'r', encoding = 'utf-8') as file:
        idx = 0
        async for line in file:
            line = line.strip()
            if line:
                try:
                    json_obj = json.loads(line)
                    json_objs.append(json_obj)
                except json.JSONDecodeError as e:
                    print(f'Line: {idx}, Error: {e}')
                except Exception as e:
                    print(f'Line: {idx}, Unexpected Error: {e}')
            idx += 1
    return json_objs

async def awrite_jsonl(path: str, json_objs: List[Any], append: bool = False):
    mode = 'a' if append else 'w'
    async with aiofiles.open(path, mode, encoding = 'utf-8') as file:
        for json_obj in json_objs:
            json_line = json.dumps(json_obj, ensure_ascii = False)
            await file.write(json_line + '\n')

def read_sampled_data(language: str):
    datas = []
    zh_dir = f'./data_raw/{language}_data_sampled/'
    for path in os.listdir(zh_dir):
        with open(os.path.join(zh_dir, path), 'r', encoding = 'utf-8') as file:
            data = json.load(file)
            data['language'] = language
            datas.append(data)
    return datas

def stable_hash(data):
    if not isinstance(data, str):
        data = str(data)
    data = data.encode('utf-8')
    return int.from_bytes(hashlib.sha256(data).digest()[:8], byteorder='big')