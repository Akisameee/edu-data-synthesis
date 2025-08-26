import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

import dspy
from dspy.datasets import HotPotQA

from models import Base_LLM, get_model
from modules.nodes import *
from modules.workflow import *
from modules.prompt_templates import *
from modules.utils import *

def evaluation_metric(data, result, trace=None):
    scores_labels = data._store['answer']
    try:
        scores_predict = extract_json(result._store['answer'])
        scores_predict = Evaluate.check_scores(scores_predict, data._store['criteria'])
        scores_predict = EvalScores([EvalScore(**score) for score in scores_predict])
    except:
        return 0
    
    corrs = {}
    for eval, scores_label in scores_labels.items():
        scores_flatten_labels: List[float] = []
        scores_flatten_predicts: List[float] = []
        if scores_label is None or scores_predict is None:
            continue
        criteria = {s.criterion for s in scores_label} & {s.criterion for s in scores_predict}
        for criterion in criteria:
            scores_flatten_labels.append(scores_label.get_score(criterion).score)
            scores_flatten_predicts.append(scores_predict.get_score(criterion).score)

        corr = 0
        for a, b in zip(scores_flatten_labels, scores_flatten_predicts):
            corr += abs(a - b) / 10
        corrs[eval] = (1 - (corr / len(scores_flatten_labels)))
    print(corrs)
    return max([corr for eval, corr in corrs.items()])

def perpare_train_datas():

    language = 'zh'
    scenarios = read_scenarios('./data/scenario', language)
    criterias = read_criterias('./data/criteria', language)  

    sub_eval_datas = read_jsonl(f'./eval_res/sub_eval_samples.jsonl')
    eval_datas = read_jsonl(f'./eval_res/eval_samples.jsonl')
    eval_datas = [
        data for data in eval_datas
        if data['id'] not in [s_data['id'] for s_data in sub_eval_datas]
    ]

    human_eval_datas = {}
    for eval_data in eval_datas:
        if not eval_data['eval'].startswith('human_'):
            continue
        if eval_data['id'] not in human_eval_datas:
            scenario = scenarios[eval_data['task']]
            messages = Messages([Message(**message) for message in eval_data['message']])
            question = evaluation_template.format(
                scenario = scenario,
                message = messages.to_json(),
                criteria = criterias[scenario['task']]
            )
            human_eval_datas[eval_data['id']] = dspy.Example({
                'question': question,
                'answer': {},
                'criteria': criterias[scenario['task']]
            })
        scores = EvalScores([EvalScore(**score) for score in eval_data['scores']])
        human_eval_datas[eval_data['id']]['answer'][eval_data['eval']] = scores
        
    return human_eval_datas

eval_datas = perpare_train_datas()

eval_models = ['qwen-max', 'deepseek-v3', 'deepseek-r1', 'gpt-4o']
eval_models = [get_model(model) for model in eval_models]
eval_model = eval_models[1]
dspy.configure(
    lm=dspy.LM(
        f'openai/{eval_model.model_name_client}',
        api_key = eval_model.api_key,
        base_url = eval_model.base_url
    )
)

trainset = [x.with_inputs('question') for id, x in eval_datas.items()]
agent = dspy.ChainOfThought("question -> answer")

tp = dspy.MIPROv2(metric=evaluation_metric, auto="light", num_threads=4)
agent = tp.compile(agent, trainset=trainset)
agent.save(f"optimized_agent.json")