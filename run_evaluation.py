import json
from tqdm import tqdm

from models import get_model
from modules.state import *
from modules.actions import *
from modules.utils import *

if __name__ == '__main__':

    gen_method = 'io_workflow'
    # gen_method = 'manual_seq_workflow'
    # gen_method = 'test_run'
    language = 'zh'

    scenarios = read_scenarios('./data/scenario', language)
    criterias = read_criterias('./data/criteria')

    eval_models = ['qwen-max', 'deepseek-v3', 'deepseek-r1', 'gpt-4o']
    eval_models = [get_model(model) for model in eval_models]    

    gen_datas = read_jsonl(f'./gen_res/{gen_method}.jsonl')

    for gen_data in tqdm(gen_datas):
        for eval_model in eval_models:

            eval_datas = read_jsonl(f'./eval_res/{gen_method}.jsonl')
            if any(
                e_d['id'] == gen_data['id'] and e_d['eval'] == eval_model.model_name
                for e_d in eval_datas
            ):
                print('repeated sample')
                continue

            scenario = scenarios[gen_data['task']]
            state = SynthesisState()
            state.set_state({
                'scenario': scenario,
                'criteria': criterias[scenario['task']],
                'message': gen_data['message']
            })
            try:
                evaluate = Evaluate()
                state = evaluate(
                    state = state,
                    llm = eval_model
                )
                eval_datas.append({
                    **gen_data,
                    'eval': eval_model.model_name,
                    'scores': state.scores
                })
            except Exception as e:
                print(str(e))
                continue
            
            eval_datas.sort(key = lambda d: int(d['id']))
            write_jsonl(f'./eval_res/{gen_method}.jsonl', eval_datas)