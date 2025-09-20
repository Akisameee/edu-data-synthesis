import dspy
from dspy.datasets import HotPotQA

from modules.models import Base_LLM, get_model
from modules.workflow import *
from modules.nodes.utils import extract_json

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

    train_dataset = EvaluationDataset('./data/eval_data/train_eval_data.jsonl')
    val_dataset = EvaluationDataset('./data/eval_data/val_eval_data.jsonl')
        
    trainset = [x.with_inputs('question') for id, x in train_dataset.items()]
    valset = [x.with_inputs('question') for id, x in val_dataset]
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