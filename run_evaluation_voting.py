import json
from tqdm import tqdm
from copy import deepcopy
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

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
    gen_method = 'sub_eval_samples'
    language = 'zh'

    scenarios = read_scenarios('./data/scenario', language)
    criterias = read_criterias('./data/criteria', language)

    eval_models = ['qwen-max', 'deepseek-v3', 'deepseek-r1', 'gpt-4o']
    eval_models = [get_model(model) for model in eval_models]    

    gen_datas = read_jsonl(f'./gen_res/{gen_method}.jsonl')

    # 创建必要的锁
    file_lock = threading.Lock()  # 用于文件读写操作
    model_locks = {model.model_name: threading.Lock() for model in eval_models}  # 每个模型一个锁

    def process_task(gen_data, eval_model, gen_method):
        # 检查是否已存在评估结果 (需要文件锁)
        with file_lock:
            eval_name = f'voting-{eval_model.model_name}'
            eval_datas = read_jsonl(f'./eval_res/{gen_method}.jsonl')
            if any(
                e_d['id'] == gen_data['id'] and e_d['eval'] == eval_name
                for e_d in eval_datas
            ): return

            eval_ress = [
                e_d for e_d in eval_datas
                if e_d['id'] == gen_data['id'] and e_d['eval'] in [
                    m.model_name for m in eval_models
                ]
            ]
            if len(eval_ress) != len(eval_models): return

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
            node = EvaluationVoting(eval_model)
            scores = node(scores_list, **eval_args)
        except Exception as e:
            print(str(e))
            return

        # 保存结果 (需要文件锁)
        eval_data = {
            **gen_data,
            'eval': eval_name,
            'scores': scores.to_json()
        }
        
        with file_lock:
            # 再次检查避免其他线程已写入相同结果
            eval_datas = read_jsonl(f'./eval_res/{gen_method}.jsonl')
            if any(
                e_d['id'] == gen_data['id'] and e_d['eval'] == eval_name
                for e_d in eval_datas
            ): return
            
            eval_datas.append(eval_data)
            eval_datas.sort(key=lambda d: d['id'])
            write_jsonl(f'./eval_res/{gen_method}.jsonl', eval_datas)

    # 创建任务列表
    tasks = []
    for gen_data in gen_datas:
        for eval_model in eval_models:
            eval_name = f'voting-{eval_model.model_name}'
            eval_datas = read_jsonl(f'./eval_res/{gen_method}.jsonl')
            if any(
                e_d['id'] == gen_data['id'] and e_d['eval'] == eval_name
                for e_d in eval_datas
            ): continue
            tasks.append((gen_data, eval_model))

    # 使用线程池执行
    with ThreadPoolExecutor(max_workers=8) as executor:  # 根据API限制调整线程数
        futures = {
            executor.submit(process_task, gen_data, eval_model, gen_method): (gen_data, eval_model)
            for gen_data, eval_model in tasks
        }
        
        # 使用tqdm显示进度
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Evaluating"):
            try:
                future.result()  # 获取结果（如有异常会在此抛出）
            except Exception as e:
                # 这里处理线程执行中未捕获的异常
                gen_data, eval_model = futures[future]
                print(f"Unhandled error in task {gen_data['id']}/{eval_model.model_name}: {str(e)}")