import json
from tqdm import tqdm
import asyncio
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

from modules.models import get_model
from modules.workflow import *
from utils import *

if __name__ == '__main__':

    messages_list, scores_labels_dict = perpare_eval_datas()

    # eval_workflow = EvaluationWorkflow()
    # eval_workflow.add_node('evaluate', Evaluate(eval_models[1]))
    # eval_workflow.add_edge('input', 'evaluate')
    # eval_workflow.add_edge('evaluate', 'output')
    # eval_workflow.save('eval_workflow.json')
    eval_workflow = EvaluationWorkflow.load('eval_workflow.json')
    score = asyncio.run(eval_workflow.evaluate(messages_list, scores_labels_dict))
    