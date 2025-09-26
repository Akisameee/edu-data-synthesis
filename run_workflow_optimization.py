import json
from tqdm import tqdm
import asyncio
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

from modules.workflow import *
from modules.optimizer import *
from modules.datas import *

if __name__ == '__main__':

    val_dataset = EvaluationDataset('./eval_res/sub_eval_samples.jsonl')
    val_dataset = val_dataset.sub_criterion(val_dataset.criteria[0].name)

    eval_workflow = EvaluationWorkflow().from_dict('./init_workflows/init_evaluation_workflow_1_1.json')
    eval_workflow.add_node('evaluate_0', Evaluate('deepseek-chat'))
    eval_workflow.add_node('evaluate_1', Evaluate('deepseek-reasoner'))
    eval_workflow.add_node('evaluate_2', Evaluate('deepseek-chat', ['web_search']))
    eval_workflow.add_node('aggregate_0', EvaluationAggregation('deepseek-chat'))
    eval_workflow.add_node('voting_0', EvaluationVoting('deepseek-chat'))
    eval_workflow.add_node('average', EvaluationAverage())
    eval_workflow.add_node('max', EvaluationMax())
    eval_workflow.add_node('min', EvaluationMin())

    eval_workflow.add_edge('input', 'evaluate_0')
    eval_workflow.add_edge('evaluate_0', 'output')
    optimizer = LocalSearch(eval_workflow, val_dataset)
    workflow = optimizer.run()
    print(workflow.to_dict())
    workflow.save()