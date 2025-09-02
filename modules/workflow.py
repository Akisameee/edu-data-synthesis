import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau
from tqdm.asyncio import tqdm_asyncio
from copy import deepcopy
import asyncio
import functools
from typing import Literal, List, Dict, Generic, TypeVar, ClassVar
from dataclasses import dataclass, field

import sys
sys.path.insert(0, '.')

from modules.models import Base_LLM, get_model
from modules.base import *
from modules.nodes import *
from modules.nodes.prompt_templates import *
from modules.utils import *

class Workflow:

    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {'output': Output()}
        self.edges: List[Tuple[str, str]] = []

    def add_node(self, name: str, node: Node) -> int:
        self.nodes[name] = node

    def pop_node(self, name: str) -> Node:
        return self.nodes.pop(name)
    
    def add_edge(self, name_a: str, name_b: str) -> None:
        if (name_a, name_b) in self.edges:
            return
        node_a, node_b = self.nodes[name_a], self.nodes[name_b]
        node_a.children.append(node_b)
        if node_b.max_indegree == None:
            node_b.parents.append(node_a)
        elif node_b.max_indegree > 0:
            if len(node_b.parents) < node_b.max_indegree:
                node_b.parents.append(node_a)
            else:
                raise ValueError(f'Exceeded maximum indegree of node {node_b.__class__}')
    
    def remove_edge(self, name_a: str, name_b: str) -> None:
        if (name_a, name_b) not in self.edges:
            return
        self.edges.remove((name_a, name_b))
        node_a, node_b = self.nodes[name_a], self.nodes[name_b]
        node_a.children.remove(node_b)
        node_b.parents.remove(node_a)

    @staticmethod
    def check_graph(
        nodes: Dict[str, Node],
        edges: List[Tuple[str, str]],
        current: Node = None
    ) -> bool:
        
        if 'output' not in nodes:
            return False
        
        for name, node in nodes.items():
            for parent in node.parents:
                pass

    def get_neighbor(self) -> 'Workflow':
        pass

    async def evaluate(self, **kwargs) -> float:
        raise NotImplementedError

    async def run(self, **kwargs) -> Messages | EvalScore:
        return await self.nodes['output'].run(**kwargs)

class EvaluationWorkflow(Workflow):

    def __init__(self) -> None:
        super().__init__()
        self.nodes: Dict[str, Node] = {'output': EvaluationOutput()}

    @staticmethod
    def calculate_correlation(
        scores_labels: List[EvalScores],
        scores_predicts: List[EvalScores],
        method: Literal['pearson', 'spearman', 'kendall'] = 'kendall'
    ):
        scores_flatten_labels: List[float] = []
        scores_flatten_predicts: List[float] = []
        for scores_label, scores_predict in zip(scores_labels, scores_predicts):
            if scores_label is None or scores_predict is None:
                continue
            criteria = {s.criterion for s in scores_label} & {s.criterion for s in scores_predict}
            for criterion in criteria:
                scores_flatten_labels.append(scores_label.get_score(criterion).score)
                scores_flatten_predicts.append(scores_predict.get_score(criterion).score)
        
        if len(scores_flatten_labels) < 2:
            return (np.nan, np.nan)
        
        if method == 'pearson':
            corr, pval = pearsonr(scores_flatten_labels, scores_flatten_predicts)
        elif method == 'spearman':
            corr, pval = spearmanr(scores_flatten_labels, scores_flatten_predicts)
        elif method == 'kendall':
            corr, pval = kendalltau(scores_flatten_labels, scores_flatten_predicts)
        else:
            raise ValueError("Method must be 'pearson', 'spearman', or 'kendall'")
        
        return (corr, pval)

    async def evaluate(
        self,
        inputs: List[dict],
        labels_dicts: List[Dict[str, EvalScores]],
        max_parallel: int = 8
    ) -> float:
        
        semaphore = asyncio.Semaphore(max_parallel)
        async def run_with_semaphore(index, input_kwargs):
            async with semaphore:
                try:
                    result = await self.run(**input_kwargs)
                    return index, result
                except Exception as e:
                    return index, None
        
        tasks = []
        for i, input_kwargs in enumerate(inputs):
            task = run_with_semaphore(i, input_kwargs)
            tasks.append(task)
        
        predicts = [None] * len(inputs)
        for task in tqdm_asyncio.as_completed(tasks, desc = 'EvalWorkflow Evaluation'):
            index, result = await task
            predicts[index] = result

        corr_1, _ = self.calculate_correlation([labels_dict['human_1'] for labels_dict in labels_dicts], predicts)
        print(f'Human_1 corr: {corr_1}')
        corr_2, _ = self.calculate_correlation([labels_dict['human_2'] for labels_dict in labels_dicts], predicts)
        print(f'Human_2 corr: {corr_2}')
        corr_3, _ = self.calculate_correlation([labels_dict['human_3'] for labels_dict in labels_dicts], predicts)
        print(f'Human_3 corr: {corr_3}')
        corr = max([corr_1, corr_2, corr_3])
        print(f'Max corr: {corr}')
        return corr

class GenerationWorkflow(Workflow):

    def __init__(self) -> None:
        super().__init__()