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
        except:
            return None
    else:
        return None