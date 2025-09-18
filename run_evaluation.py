import json
from tqdm import tqdm
import asyncio
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

from modules.workflow import *
from utils import *

if __name__ == '__main__':

    # val_dataset = EvaluationDataset('./data/eval_data/val_eval_data.jsonl')
    val_dataset = EvaluationDataset('./eval_res/sub_eval_samples.jsonl')
    val_dataset = val_dataset.sub_criterion(val_dataset.criteria[4].name)

    eval_workflow = EvaluationWorkflow()
    # eval_workflow.add_node('evaluate_0', Evaluate('deepseek-v3'))
    eval_workflow.add_node('evaluate_0', Evaluate('deepseek-v3', ['web_search']))
    eval_workflow.add_edge('input', 'evaluate_0')
    eval_workflow.add_edge('evaluate_0', 'output')
    score = asyncio.run(eval_workflow.evaluate(val_dataset))
    