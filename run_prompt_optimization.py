import json
from tqdm import tqdm
import asyncio
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

from modules.workflow import *
from modules.optimizer import *
from modules.datas import *

if __name__ == '__main__':

    val_dataset = EvaluationDataset('./data/eval_data/train_eval_data.jsonl')
    val_dataset = val_dataset.sub_criterion(val_dataset.criteria[0].name)

    optimizer = FewshotSampleOptimizer(
        Evaluate('deepseek-chat')
    )
    node = optimizer.run()
    print(node.to_dict())