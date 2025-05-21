import os
import json
import pandas as pd
from typing import Literal

import sys
sys.path.insert(0, '.')

from data.utils import yield_json_files

levels = ['primary', 'junior', 'senior', 'undergraduate', 'graduate']
subjects = ['computer_science', 'chemistry', 'history', 'geography', 'math', 'physics', 'biology', 'science', 'english', 'chinese', 'political_science']

class Sampler():

    def __init__(
        self,
        data_dir: str,
        scope: str = None
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
                        'data': data,
                        'used_scopes': []
                    })

        self.datas = pd.DataFrame(datas)

    def set_scope(self, new_scope: str):

        self.scope = new_scope

    def get_question_database_info(self) -> str:

        df = self.datas
        df = df[~df['used_scopes'].apply(lambda scopes: self.scope in scopes)]
        df_info = ['questions database info (level/subject/type)']

        counts = df.groupby(['level', 'subject', 'type']).size().reset_index(name='count')

        last_level = None
        last_subject = None

        for _, row in counts.iterrows():
            level = row['level']
            subject = row['subject']
            type_ = row['type']
            count = row['count']

            if level != last_level:
                level_count = df[df['level'] == level].shape[0]
                df_info.append(f'-{level}: {level_count}')
                last_level = level
                last_subject = None

            if subject != last_subject:
                subject_count = df[(df['level'] == level) & (df['subject'] == subject)].shape[0]
                df_info.append(f'\t-{subject}: {subject_count}')
                last_subject = subject

            df_info.append(f'\t\t-{type_}: {count}')

        return '\n'.join(df_info)
    
    def sample_question(
        self,
        level: str = None,
        subject: str = None,
        type_: str = None
    ) -> str:
        
        df = self.datas
        if level:
            levels = df['level'].unique()
            if level not in levels:
                raise ValueError(
                    f'Invalid level \'{level}\', level must be one of {levels}.'
                )
            df = df[df['level'] == level]

        if subject:
            subjects = df['subject'].unique()
            if subject not in subjects:
                raise ValueError(
                    f'Invalid subject \'{subject}\', subject must be one of {subjects}.'
                )
            df = df[df['subject'] == subject]

        if type_:
            types = df['type'].unique()
            if type_ not in types:
                raise ValueError(
                    f'Invalid type \'{type_}\', type must be one of {types}.'
                )
            df = df[df['type'] == type_]

        df = df[~df['used_scopes'].apply(lambda scopes: self.scope in scopes)]

        sampled_row = df.sample(n=1).iloc[0]
        sampled_row['used_scopes'].append(self.scope)

        sample = {
            'level': sampled_row['level'],  
            'subject': sampled_row['subject'],  
            'type': sampled_row['type'],
            **sampled_row['data']
        }

        return json.dumps(sample, ensure_ascii = False)

if __name__ == '__main__':

    data_dir = './data/zh'
    sampler = Sampler(data_dir, scope = 'correction')

    print(sampler.get_question_database_info())
    print(sampler.sample_question(
        'junior',
        'math'
    ))