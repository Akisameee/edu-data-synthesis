import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from copy import deepcopy
import asyncio
import functools
from typing import Literal, List, Dict, Set, Generic, TypeVar, ClassVar
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
        self.nodes: Dict[str, Node] = {'input': Input(), 'output': Output()}

    def add_node(self, name: str, node: Node) -> bool:
        if node in self.nodes.values():
            return False
        else:
            self.nodes[name] = node
            return True

    def pop_node(self, name: str) -> Node:
        return self.nodes.pop(name)
    
    def _get_node_name(self, node: Node) -> Optional[str]:
        for key, value in self.nodes.items():
            if value == node:
                return key
        return None
        
    @property
    def edges(self) -> Set[Tuple[str, str]]:
        edges = set()
        for name, node in self.nodes.items():
            for parent in node.parents:
                parent_name = self._get_node_name(parent)
                if node in parent.children:
                    edges.add((parent_name, name))
                else:
                    raise ValueError(f'Invalid edge: {parent_name} -> {name}')
            for child in node.children:
                child_name = self._get_node_name(child)
                if node in child.parents:
                    edges.add((name, child_name))
                else:
                    raise ValueError(f'Invalid edge: {name} -> {child_name}')
        return edges
    
    def add_edge(self, name_a: str, name_b: str) -> bool:
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
        node_a, node_b = self.nodes[name_a], self.nodes[name_b]
        node_a.children.remove(node_b)
        node_b.parents.remove(node_a)

    def check_graph(self) -> bool:
        
        if 'input' not in self.nodes or 'output' not in self.nodes:
            return False
        return True
        
        print(self.edges)
        # for name, node in self.nodes.items():
        #     if not node.check_parent():
        #         return False

    def get_neighbor(self) -> 'Workflow':
        pass

    async def evaluate(self, messages: Messages) -> float:
        raise NotImplementedError

    async def run(self, messages: Messages) -> Messages:
        if not self.check_graph():
            raise RuntimeError(
                'Invalid Workflow:\n' + f'Nodes: {self.nodes.keys()}' + f'Edges: {self.edges}'
            )
        return await self.nodes['output'].run(messages)

class EvaluationWorkflow(Workflow):

    def __init__(self) -> None:
        super().__init__()
        self.nodes: Dict[str, Node] = {
            'input': EvaluationInput(),
            'output': EvaluationOutput()
        }

    @staticmethod
    def calculate_correlation(
        scores_labels: List[EvalScores],
        scores_predicts: List[EvalScores],
        method: Literal['pearson', 'spearman', 'kendall'] = 'kendall'
    ) -> Tuple[float, float]:
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
        messages_list: List[Messages],
        scores_labels_list: Dict[str, List[EvalScores]],
        max_parallel: int = 8
    ) -> float:
        
        semaphore = asyncio.Semaphore(max_parallel)
        async def run_with_semaphore(index, inputs):
            async with semaphore:
                try:
                    messages = await self.run(inputs)
                    return index, messages
                except Exception as e:
                    return index, None
        
        tasks = []
        for i, messages in enumerate(messages_list):
            task = run_with_semaphore(i, messages)
            tasks.append(task)
        
        scores_predicts = [None] * len(messages_list)
        for task in tqdm_asyncio.as_completed(tasks, desc = 'EvalWorkflow Evaluation'):
            index, messages = await task
            scores_predicts[index] = messages.scores

        # scores_predicts = []
        # for messages in tqdm(messages_list):
        #     messages = await self.run(messages)
        #     scores_predicts.append(messages.scores)

        corrs = []
        for eval, scores_labels in scores_labels_list.items():
            corr, _ = self.calculate_correlation(scores_labels, scores_predicts)
            corrs.append(corr)
            print(f'{eval} correlation: {corr}')
        
        corr = max(corrs)
        print(f'max correlation: {corr}')
        return corr

class GenerationWorkflow(Workflow):

    def __init__(self) -> None:
        super().__init__()