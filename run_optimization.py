import json
from tqdm import tqdm
import asyncio
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

from modules.models import get_model
from modules.workflow import *
from modules.optimizer import *
from utils import *

if __name__ == '__main__':

    messages_list, scores_labels_dict = perpare_eval_datas()

    eval_workflow = EvaluationWorkflow()
    eval_workflow.add_node('evaluate_0', Evaluate(get_model('deepseek-v3')))
    eval_workflow.add_node('evaluate_1', Evaluate(get_model('deepseek-v3')))
    eval_workflow.add_node('aggregate_0', EvaluationAggregation(get_model('deepseek-v3')))
    eval_workflow.add_edge('input', 'evaluate_0')
    eval_workflow.add_edge('evaluate_0', 'output')
    optimizer = LocalSearch(eval_workflow)
    print(optimizer.get_mutation_ops(optimizer.workflow, []))
    print(('input', 'output') in eval_workflow.edges)