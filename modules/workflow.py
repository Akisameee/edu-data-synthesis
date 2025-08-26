import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau
from tqdm import tqdm
from copy import deepcopy
import random
import functools
from typing import Literal, List, Dict, Generic, TypeVar, ClassVar
from dataclasses import dataclass, field

import sys
sys.path.insert(0, '.')

from models import Base_LLM, get_model
from modules.nodes import *
from modules.prompt_templates import *
from modules.utils import *


class Workflow:

    def __init__(self) -> None:
        self.output_node: Output = Output()
        self.nodes: Dict[int, Node] = {0: self.output_node}
        self.edges: List[Tuple[int, int]] = []

    def add_node(self, node: Node) -> int:
        idx = 0
        while idx in self.nodes: idx += 1
        self.nodes[idx] = node
        return idx

    def pop_node(self, idx: int) -> Node:
        return self.nodes.pop(idx)
    
    def add_edge(self, idx_a: int, idx_b: int) -> None:
        if (idx_a, idx_b) in self.edges:
            return
        node_a, node_b = self.nodes[idx_a], self.nodes[idx_b]
        node_a.children.append(node_b)
        if node_b.max_indegree == None:
            node_b.parents.append(node_a)
        elif node_b.max_indegree > 0:
            if len(node_b.parents) < node_b.max_indegree:
                node_b.parents.append(node_a)
            else:
                raise ValueError(f'Exceeded maximum indegree of node {node_b.__class__}')
    
    def remove_edge(self, idx_a: int, idx_b: int) -> None:
        if (idx_a, idx_b) not in self.edges:
            return
        self.edges.remove((idx_a, idx_b))
        node_a, node_b = self.nodes[idx_a], self.nodes[idx_b]
        node_a.children.remove(node_b)
        node_b.parents.remove(node_a)

    @staticmethod
    def check_dag(nodes: List[Node]) -> List[Node]:
        pass

    def get_neighbor(self) -> 'Workflow':
        pass

    def evaluate(self, **kwargs) -> float:
        raise NotImplementedError

    def run(self, **kwargs) -> Messages | EvalScore:
        return self.output_node.run(**kwargs)

class EvaluationWorkflow(Workflow):

    def __init__(self) -> None:
        super().__init__()

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

    def evaluate(
        self,
        inputs: List[dict],
        labels_dicts: List[Dict[str, EvalScores]]
    ) -> float:

        predicts = []
        for input_kwargs in tqdm(inputs, desc = 'EvalWorkflow Evaluation'):
            try:
                predict = self.run(**input_kwargs)
                # print(f"{input_kwargs['id']}: {[score.score for score in predict]}\n")
            except Exception as e:
                predict = None
            predicts.append(predict)

        corr_1 = self.calculate_correlation([labels_dict['human_1'] for labels_dict in labels_dicts], predicts)[0]
        print(f'Human_1 corr: {corr_1}')
        corr_2 = self.calculate_correlation([labels_dict['human_2'] for labels_dict in labels_dicts], predicts)[0]
        print(f'Human_2 corr: {corr_2}')
        corr_3 = self.calculate_correlation([labels_dict['human_3'] for labels_dict in labels_dicts], predicts)[0]
        print(f'Human_3 corr: {corr_3}')
        corr = max([corr_1, corr_2, corr_3])
        print(f'Max corr: {corr}')
        return corr

class GenerationWorkflow(Workflow):

    def __init__(self) -> None:
        super().__init__()