import os
import json
import pandas as pd
from typing import Literal

import sys
sys.path.insert(0, '.')

from data.utils import yield_json_files

levels = ['primary', 'junior', 'senior', 'undergraduate', 'graduate']
subjects = ['computer_science', 'chemistry', 'history', 'geography', 'math', 'physics', 'biology', 'science', 'english', 'chinese', 'political_science']
types = []

class SamplerModule():

    def __init__(
        self,
        data_dir: str,
        scope: str
    ) -> None:
        
        self.scope = scope
        datas = []

        for json_path in yield_json_files(data_dir):
            with open(json_path, 'r', encoding = 'utf-8') as file:
                for line in file.readlines():
                    data = json.loads(line)

                    level = data.pop('level')
                    subject = data.pop('subject')  
                    type_ = data.pop('type')

                    datas.append({
                        'level': level,
                        'subject': subject,
                        'type': type_,
                        'meta_data': data,
                        'used_scopes': []
                    })

        self.datas = pd.DataFrame(datas)

    def change_scope(self, new_scope: str):

        self.scope = new_scope

    def get_database_info(self) -> str:

        pass
    
    def sample_from_database(
        self,
        level: str = None,
        subject: str = None,
        type_: str = None
    ) -> dict:
        
        df = self.datas
        if level:
            if level not in df['level'].unique():
                raise ValueError(
                    f'Invalid level \'{level}\', level must be one of {df['level'].unique()}.'
                )
            df = df[df['level'] == level]

        if subjects:
            if subject not in df['subject'].unique():
                raise ValueError(
                    f'Invalid subject \'{subject}\', subject must be one of {df['subject'].unique()}.'
                )
            df = df[df['subject'] == subject]

        if type_:
            if type_ not in df['type'].unique():
                raise ValueError(
                    f'Invalid type \'{type_}\', type must be one of {df['type'].unique()}.'
                )
            df = df[df['type'] == type_]

        df = df[~df['used_scopes'].apply(lambda scopes: self.scope in scopes)]

        sampled_row = df.sample(n=1).iloc[0]
        sampled_row['used_scopes'].append(self.scope)

        return {
            'level': sampled_row['level'],  
            'subject': sampled_row['subject'],  
            'type': sampled_row['type'],
            **sampled_row['meta_data']
        }

if __name__ == '__main__':

    data_dir = './data/zh'
    sampler = SamplerModule(data_dir, scope = 'correction')

    print(sampler.sample_from_database(
        'junior',
        'math'
    ))