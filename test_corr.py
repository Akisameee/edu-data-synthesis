import json
from tqdm import tqdm
import asyncio
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

from modules.models import get_model
from modules.workflow import *
from utils import *

if __name__ == '__main__':

    val_dataset = EvaluationDataset('./data/eval_data/val_eval_data.jsonl')
    # val_dataset = EvaluationDataset('./eval_res/sub_eval_samples.jsonl')
    val_dataset = val_dataset.sub_criterion(val_dataset.criteria[0].name)

    print(EvaluationWorkflow.calculate_correlation(
        val_dataset.labels['human_1'], val_dataset.labels['human_2']
    ))
    print(EvaluationWorkflow.calculate_correlation(
        val_dataset.labels['human_2'], val_dataset.labels['human_3']
    ))
    print(EvaluationWorkflow.calculate_correlation(
        val_dataset.labels['human_1'], val_dataset.labels['human_3']
    ))
    