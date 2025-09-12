import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

from modules.models import get_model
from modules.utils import *
from modules.base import *

def perpare_eval_datas() -> Tuple[List[Messages], Dict[str, List[EvalScores]]]:
    language = 'zh'

    scenarios = read_scenarios('./data/scenario', language)
    criterias = read_criterias('./data/criteria', language)

    eval_models = ['qwen-max', 'deepseek-v3', 'deepseek-r1', 'gpt-4o']
    eval_models = [get_model(model) for model in eval_models]    

    eval_datas = read_jsonl('./eval_res/sub_eval_samples.jsonl')

    human_eval_datas = {}
    evals = []
    for eval_data in eval_datas:

        if not eval_data['eval'].startswith('human_'):
            continue
        
        if eval_data['id'] not in human_eval_datas:
            scenario = scenarios[eval_data['task']]
            messages = Messages(eval_data['message'])
            messages.source = eval_data['gen']
            messages.metadata = {
                'id': eval_data['id'],
                'task': eval_data['task'],
                'scenario': scenario,
                'criteria': criterias[scenario['task']],
            }
            human_eval_datas[eval_data['id']] = {'messages': messages}
        if eval_data['eval'] not in evals:
            evals.append(eval_data['eval'])
        human_eval_datas[eval_data['id']][eval_data['eval']] = EvalScores(eval_data['scores'])

    messages_list = []
    scores_labels_dict = {eval: [] for eval in evals}
    for id, data in human_eval_datas.items():
        messages_list.append(data['messages'])
        for eval in evals:
            scores_labels_dict[eval].append(data[eval])

    return messages_list, scores_labels_dict