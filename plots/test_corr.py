import json
from tqdm import tqdm
import asyncio

import sys
sys.path.insert(0, '..')

from modules.workflow import *

if __name__ == '__main__':

    val_dataset = EvaluationDataset('./data/eval_data/val_eval_data.jsonl')
    # val_dataset = EvaluationDataset('./eval_res/sub_eval_samples.jsonl')
    for name in val_dataset.criteria.names:
        sub_dataset = val_dataset.sub_criterion(name)

        corr_12, _ = EvaluationWorkflow.calculate_correlation(
            sub_dataset.labels['human_1'], sub_dataset.labels['human_2']
        )
        corr_23, _ = EvaluationWorkflow.calculate_correlation(
            sub_dataset.labels['human_2'], sub_dataset.labels['human_3']
        )
        corr_13, _ = EvaluationWorkflow.calculate_correlation(
            sub_dataset.labels['human_1'], sub_dataset.labels['human_3']
        )
        print(f'criterion: {name}, corrs: {[corr_12, corr_23, corr_13]}')