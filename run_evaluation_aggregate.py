import json
from tqdm import tqdm
from copy import deepcopy

from models import get_model
from modules.state import *
# from modules.actions import *
from modules.nodes import *
from modules.utils import *

if __name__ == '__main__':

    # gen_method = 'function_calling_test'
    gen_method = 'io_workflow'
    # gen_method = 'manual_seq_workflow'
    # gen_method = 'test_run'
    gen_method = 'eval_samples'
    language = 'zh'

    scenarios = read_scenarios('./data/scenario', language)
    criterias = read_criterias('./data/criteria', language)

    eval_models = ['qwen-max', 'deepseek-v3', 'deepseek-r1', 'gpt-4o']
    eval_models = [get_model(model) for model in eval_models]    

    gen_datas = read_jsonl(f'./gen_res/{gen_method}.jsonl')

    # for gen_data in tqdm(gen_datas):
    #     for eval_model in eval_models:

    #         eval_datas = read_jsonl(f'./eval_res/{gen_method}.jsonl')
    #         if any(
    #             e_d['id'] == gen_data['id'] and e_d['eval'] == eval_model.model_name
    #             for e_d in eval_datas
    #         ):
    #             # print('repeated sample')
    #             continue

    #         scenario = scenarios[gen_data['task']]
    #         state = SynthesisState()
    #         state.set_state({
    #             'scenario': scenario,
    #             'criteria': criterias[scenario['task']],
    #             'message': gen_data['message']
    #         })
    #         if 'scores' in gen_data.keys() and \
    #             eval_model.model_name in gen_data['scores'].keys():
    #             state.scores = gen_data['scores'][eval_model.model_name]
    #         else:
    #             try:
    #                 evaluate = Evaluate()
    #                 state = evaluate(state, eval_model)
    #             except Exception as e:
    #                 print(str(e))
    #                 continue
            
    #         eval_data = {
    #             **gen_data,
    #             'eval': eval_model.model_name
    #         }
    #         eval_data['scores'] = state.scores
    #         eval_datas.append(eval_data)

    #         eval_datas.sort(key = lambda d: int(d['id']))
    #         write_jsonl(f'./eval_res/{gen_method}.jsonl', eval_datas)

    for gen_data in tqdm(gen_datas):
        for eval_model in eval_models:

            eval_datas = read_jsonl(f'./eval_res/{gen_method}.jsonl')
            if any(
                e_d['id'] == gen_data['id'] and e_d['eval'] == eval_model.model_name
                for e_d in eval_datas
            ): continue

            scenario = scenarios[gen_data['task']]
            eval_args = {
                'scenario': scenario,
                'criteria': criterias[scenario['task']]
            }
            messages = Messages([Message(**message) for message in gen_data['message']])

            try:
                node = Evaluate(eval_model)
                scores = node(messages, **eval_args)
            except Exception as e:
                print(str(e))
                continue

            eval_data = {
                **gen_data,
                'eval': eval_model.model_name,
                'scores': scores.to_json()
            }
            eval_datas.append(eval_data)

            eval_datas.sort(key = lambda d: int(d['id']))
            write_jsonl(f'./eval_res/{gen_method}.jsonl', eval_datas)

    for gen_data in tqdm(gen_datas):
        for eval_model in eval_models:
            
            eval_name = f'aggregate-{eval_model.model_name}'
            eval_datas = read_jsonl(f'./eval_res/{gen_method}.jsonl')
            if any(
                e_d['id'] == gen_data['id'] and e_d['eval'] == eval_name
                for e_d in eval_datas
            ): continue
            eval_ress = [
                e_d for e_d in eval_datas
                if e_d['id'] == gen_data['id'] and e_d['eval'] in [
                    m.model_name for m in eval_models
                ]
            ]
            if len(eval_ress) != len(eval_models): continue

            scenario = scenarios[gen_data['task']]
            eval_args = {
                'scenario': scenario,
                'criteria': criterias[scenario['task']]
            }
            scores_list = [
                EvalScores([
                    EvalScore(**score)
                    for score in eval_res['scores']
                ])
                for eval_res in eval_ress
            ]
            messages = Messages([Message(**message) for message in gen_data['message']])
            for scores in scores_list: scores.messages = messages

            try:
                node = EvaluationAggregation(eval_model)
                scores = node(scores_list, **eval_args)
            except Exception as e:
                print(str(e))
                continue

            eval_data = {
                **gen_data,
                'eval': eval_name,
                'scores': scores.to_json()
            }
            eval_datas.append(eval_data)

            eval_datas.sort(key = lambda d: d['id'])
            write_jsonl(f'./eval_res/{gen_method}.jsonl', eval_datas)