import json
from tqdm import tqdm
from copy import deepcopy
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

from models import get_model
from modules.nodes import *
from modules.workflow import *
from modules.utils import *

if __name__ == '__main__':

    gen_method = 'function_calling_test'
    # gen_method = 'io_workflow'
    # gen_method = 'manual_seq_workflow'
    gen_method = 'eval_samples'
    gen_method = 'sub_eval_samples'
    language = 'zh'

    scenarios = read_scenarios('./data/scenario', language)
    criterias = read_criterias('./data/criteria', language)

    eval_models = ['qwen-max', 'deepseek-v3', 'deepseek-r1', 'gpt-4o']
    eval_models = [get_model(model) for model in eval_models]    

    eval_datas = read_jsonl(f'./eval_res/{gen_method}.jsonl')

    human_eval_datas = {}
    for eval_data in eval_datas:

        if not eval_data['eval'].startswith('human_'):
            continue

        # if eval_data['task'] in [d[0]['task'] for id, d in human_eval_datas.items()]:
        #     continue
        
        if eval_data['id'] not in human_eval_datas:
            scenario = scenarios[eval_data['task']]
            human_eval_datas[eval_data['id']] = (
                {
                    'id': eval_data['id'],
                    'task': eval_data['task'],
                    'messages': Messages([Message(**message) for message in eval_data['message']]),
                    'scenario': scenario,
                    'criteria': criterias[scenario['task']],
                },
                {}
            )
        scores = EvalScores([EvalScore(**score) for score in eval_data['scores']])
        human_eval_datas[eval_data['id']][1][eval_data['eval']] = scores

    eval_inputs = [data[0] for id, data in human_eval_datas.items()]
    eval_labels_dict = [data[1] for id, data in human_eval_datas.items()]

    eval_workflow = EvaluationWorkflow()
    eval_workflow.add_node(Evaluate(eval_models[0]))
    eval_workflow.add_edge(1, 0)
    eval_workflow.evaluate(eval_inputs, eval_labels_dict)
    