import json
from tqdm import tqdm
import asyncio
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

from modules.models import get_model
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
    evals = []
    for eval_data in eval_datas:

        if not eval_data['eval'].startswith('human_'):
            continue

        # if eval_data['task'] in [d[0]['task'] for id, d in human_eval_datas.items()]:
        #     continue
        
        if eval_data['id'] not in human_eval_datas:
            scenario = scenarios[eval_data['task']]
            messages = Messages(eval_data['message'])
            messages.source = eval_data['gen']
            messages.meta_data = {
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

    eval_workflow = EvaluationWorkflow()
    eval_workflow.add_node('evaluate', Evaluate(eval_models[1]))
    eval_workflow.add_edge('input', 'evaluate')
    eval_workflow.add_edge('evaluate', 'output')
    score = asyncio.run(eval_workflow.evaluate(messages_list, scores_labels_dict))
    