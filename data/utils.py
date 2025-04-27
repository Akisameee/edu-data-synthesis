import os

def yield_json_files(root_dir):

    for walk_res in os.walk(root_dir):
        for filename in walk_res[2]:
            file_path = os.path.join(walk_res[0], filename)
            if file_path.endswith('.json') or file_path.endswith('.jsonl'):
                yield file_path