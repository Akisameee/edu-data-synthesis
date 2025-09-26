import re
import json
import inspect
import functools
from typing import Callable
from tqdm import tqdm

from modules.base import *

def extract_json(response: str):
    match = re.search(r'```json\s*(.*)\s*```', response, re.DOTALL)
    if match: json_str = match.group(1)
    else: json_str = response
    try:
        json_obj = json.loads(json_str)
        return json_obj 
    except Exception as e:
        try:
            json_obj = fix_json_close(json_str)
            return json_obj
        except:
            raise ValueError(f'[JSON Parse Error] {str(e)}.')

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

def check_scores(
    scores: List[Dict[str, Any]] | Dict[str, Any],
    criteria: Criteria
) -> EvalScores:
    if isinstance(scores, dict): scores = [scores]
    extra_criteria = []
    for score in scores:
        criterion = [c.name for c in criteria if score['criterion'] in c.name]
        if len(criterion) > 1:
            invalid_criteria = score['criterion']
            raise ValueError(f'[Score Parse Error] Invalid criteria: {invalid_criteria}.')
        if len(criterion) == 0:
            extra_criteria.append(score['criterion'])
            continue
        score['criterion'] = criterion[0]

        value = score['score']
        if not isinstance(value, (int, float)):
            raise ValueError(f'[Score Parse Error] Invalid score value: {value}.')
    
    scores = [score for score in scores if score['criterion'] not in extra_criteria]
        
    if set(score['criterion'] for score in scores) != \
        set(c.name for c in criteria):
        invalid_criteria = [score['criterion'] for score in scores]
        required_criteria = [c.name for c in criteria]
        raise ValueError(f'[Score Parse Error] Invalid criteria: {invalid_criteria}, required: {required_criteria}.')
    
    return EvalScores(scores)

def retry(max_attempt: int = 3, verbose: bool = False) -> Callable:
    def decorator(func: Callable) -> Callable:
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                n_attempt = 0
                while n_attempt < max_attempt:
                    try: return await func(*args, **kwargs)
                    except Exception as e:
                        n_attempt += 1
                        if verbose: tqdm.write(f"Attempt {n_attempt}/{max_attempt} failed: {e}")
                        if n_attempt == max_attempt: raise e
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                n_attempt = 0
                while n_attempt < max_attempt:
                    try: return func(*args, **kwargs)
                    except Exception as e:
                        n_attempt += 1
                        if verbose: tqdm.write(f"Attempt {n_attempt}/{max_attempt} failed: {e}")
                        if n_attempt == max_attempt: raise e
            return sync_wrapper
    return decorator