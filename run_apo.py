import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

import dspy
from dspy.datasets import HotPotQA

from modules.models import Base_LLM, get_model
from modules.workflow import *

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
        scores_label = EvalScores([EvalScore(**score) for score in scores_label])
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

def perpare_datas():

    language = 'zh'
    scenarios = read_scenarios('./data/scenario', language)
    criterias = read_criterias('./data/criteria', language)  

    sub_eval_datas = read_jsonl(f'./eval_res/sub_eval_samples.jsonl')
    eval_datas = read_jsonl(f'./eval_res/eval_samples.jsonl')

    val_ids = [s_data['id'] for s_data in sub_eval_datas]

    train_datas = {}
    val_datas = {}
    for eval_data in eval_datas:
        if not eval_data['eval'].startswith('human_'):
            continue
        
        if eval_data['id'] in val_ids:
            datas = val_datas
        else:
            datas = train_datas

        if eval_data['id'] not in datas:
            scenario = scenarios[eval_data['task']]
            messages = Messages([Message(**message) for message in eval_data['message']])
            question = evaluation_template.format(
                scenario = scenario,
                message = messages.to_json(),
                criteria = criterias[scenario['task']]
            )
            datas[eval_data['id']] = dspy.Example({
                'question': question,
                'answer': {},
                'criteria': criterias[scenario['task']]
            })
        datas[eval_data['id']]['answer'][eval_data['eval']] = eval_data['scores']
        
    trainset = [x.with_inputs('question') for id, x in train_datas.items()]
    valset = [x.with_inputs('question') for id, x in val_datas.items()]
    return trainset, valset

trainset, valset = perpare_datas()

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

agent = dspy.ChainOfThought("question -> answer")

tp = dspy.MIPROv2(metric=evaluation_metric, auto="light", num_threads=8, verbose = True)
agent = tp.compile(
    agent,
    trainset = trainset,
    valset = valset,
    provide_traceback = True
)
agent.save(f"optimized_agent.pkl")
# agent.load("optimized_agent.pkl")
# print(agent)