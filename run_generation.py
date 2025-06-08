import json
from tqdm import tqdm

from modules.planner import Planner
from modules.utils import *

io_workflow = [
    ('1', 'user_generate', {'model_name': 'deepseek-v3'}),
    ('2', 'assistant_generate', {'model_name': 'deepseek-v3'}),
    ('3', 'output', {})
]

manual_seq_workflow = [
    ('1', 'user_generate', {'model_name': 'deepseek-v3'}),
    ('2', 'assistant_generate', {'model_name': 'deepseek-v3'}),
    ('3', 'review', {'model_name': 'deepseek-r1'}),
    ('4', 'refine', {'model_name': 'deepseek-v3'}),
    ('6', 'output', {})
]

if __name__ == '__main__':

    # gen_method = 'function_calling_test'
    gen_method = 'io_workflow'
    # gen_method = 'manual_seq_workflow'
    # gen_method = 'test_run'
    language = 'zh'

    scenarios = read_scenarios('./data/scenario', language)
    criterias = read_criterias('./data/criteria')

    # models = ['qwen2.5-7b-instruct', 'qwen2.5-14b-instruct', 'qwen-max', 'deepseek-v3', 'deepseek-r1']
    # models = ['deepseek-v3', 'deepseek-r1', 'qwen2.5-7b-instruct', 'qwen2.5-14b-instruct', 'qwen-max']
    models = ['deepseek-v3', 'deepseek-r1', 'qwen-max', 'gpt-4o']
    planner = Planner(models)

    sampled_datas = read_jsonl(f'./data_raw/{language}_data_sampled.jsonl')

    for sampled_data in tqdm(sampled_datas):

        gen_datas = read_jsonl(f'./gen_res/{gen_method}.jsonl')

        if any(g_d['id'] == sampled_data['id'] for g_d in gen_datas):
            continue

        scenario = scenarios[sampled_data['task']]
        init_state = {
            'scenario': scenario,
            'criteria': criterias[scenario['task']],
            'meta_data': sampled_data['meta_data']
        }
        try:
            gen_state, trajectory = planner.run_seq_workflow(
                init_state, io_workflow
            )
            # gen_state, trajectory = planner.run_function_calling(
            #     init_state,
            #     max_tool_call = 20
            # )
            tool_calls = [
                {
                    'name': action['tool_calls'][0].function.name,
                    'arguments': json.loads(action['tool_calls'][0].function.arguments)
                } for action in trajectory
                if action['role'] == 'assistant' and action['tool_calls']
            ]
            
            gen_data = {
                **sampled_data,
                'message': gen_state.message,
                'gen': gen_method,
                'tool_calls': tool_calls,
                'cost': gen_state.cost
            }
            if gen_state.scores:
                gen_data['scores'] = gen_state.scores
            gen_datas.append(gen_data)
        except Exception as e:
            tqdm.write(str(e))
            continue
        
        gen_datas.sort(key = lambda d: int(d['id']))
        write_jsonl(f'./gen_res/{gen_method}.jsonl', gen_datas)