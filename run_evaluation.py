import json
from tqdm import tqdm
import asyncio

from modules.workflow import *

if __name__ == '__main__':

    val_dataset = EvaluationDataset('./data/eval_data/val_eval_data.jsonl')
    # val_dataset = EvaluationDataset('./eval_res/sub_eval_samples.jsonl')
    val_dataset = val_dataset.sub_criterion(val_dataset.criteria[0].name)

    eval_workflow = EvaluationWorkflow()
    eval_workflow.add_node('evaluate_0', Evaluate('deepseek-chat', cache = True))
    # eval_workflow.add_node('evaluate_0', Evaluate('deepseek-chat', [], [
    #     {'id': 'en_question_37_model_0.json', 'eval': 'human_3'},
    #     {'id': 'zh_question_57_model_4.json', 'eval': 'human_3'}
    # ]))
    # eval_workflow.add_node('evaluate_0', Evaluate('deepseek-chat', ['web_search']))
    eval_workflow.add_edge('input', 'evaluate_0')
    eval_workflow.add_edge('evaluate_0', 'output')
    eval_workflow.save('test_workflow.json')
    score, cost, messages_predicts = asyncio.run(eval_workflow.evaluate(val_dataset))
    