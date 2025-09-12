import json
from tqdm import tqdm
import asyncio
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

from modules.workflow import *
from modules.optimizer import *
from modules.datas import *
from utils import *

if __name__ == '__main__':

    val_dataset = EvaluationDataset('./eval_res/sub_eval_samples.jsonl')
    for c in val_dataset.criteria:
        sub_dataset = val_dataset.sub_criterion(c.name)

    eval_workflow = EvaluationWorkflow()
    eval_workflow.add_node('evaluate_0', Evaluate('deepseek-v3'))
    eval_workflow.add_node('evaluate_1', Evaluate('deepseek-r1'))
    eval_workflow.add_node('aggregate_0', EvaluationAggregation('deepseek-v3'))
    eval_workflow.add_node('voting_0', EvaluationVoting('deepseek-v3'))
    eval_workflow.add_node('average', EvaluationAverage())
    eval_workflow.add_node('max', EvaluationMax())
    eval_workflow.add_node('min', EvaluationMin())

    eval_workflow.add_edge('input', 'evaluate_0')
    eval_workflow.add_edge('evaluate_0', 'output')
    optimizer = LocalSearch(eval_workflow, val_dataset)
    # print(optimizer.get_mutation_ops(optimizer.workflow, []))
    # print(('input', 'output') in eval_workflow.edges)
    workflow = optimizer.run()
    print(workflow.to_dict())
    workflow.save()