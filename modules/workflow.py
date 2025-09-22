import numpy as np
import pickle
import collections
import heapq
from scipy.stats import spearmanr, pearsonr, kendalltau
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from copy import deepcopy
import asyncio
from typing import Literal, List, Dict, Set, Generic, TypeVar, ClassVar, overload
from dataclasses import dataclass, field

import sys
sys.path.insert(0, '.')

from modules.models import Base_LLM, get_model
from modules.base import *
from modules.nodes import *
from modules.nodes.prompt_templates import *
from modules.datas import *
from modules.utils import *

class Workflow:

    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Tuple[str, str]] = []
        self._sub_nec: 'Workflow' = None

    def add_node(self, name: str, node: Node) -> None:
        if node.name in self.nodes:
            raise ValueError(f'Failed to add a repeated node {name}({node.__class__}).')
        node.name = name
        self.nodes[name] = node
        self._sub_nec = None

    def pop_node(self, name: str) -> Node:
        for edge in self.edges:
            if edge[0] == name or edge[1] == name:
                self.remove_edge(*edge)
        self._sub_nec = None
        return self.nodes.pop(name)
    
    def parents(self, node: str | Node) -> List[str]:
        if isinstance(node, Node): node = node.name
        return [edge[0] for edge in self.edges if edge[1] == node]
    
    def children(self, node: str | Node) -> List[str]:
        if isinstance(node, Node): node = node.name
        return [edge[1] for edge in self.edges if edge[0] == node]
    
    def indegree(self, node: str | Node) -> int:
        return len(self.parents(node))
    
    def outdegree(self, node: str | Node) -> int:
        return len(self.children(node))
    
    @overload
    def add_edge(self, node_a: str, node_b: str) -> None: ...
    @overload
    def add_edge(self, node_a: Node, node_b: Node) -> None: ...
    def add_edge(self, a: str | Node, b: str | Node) -> None:
        if isinstance(a, str):
            name_a, name_b = a, b
            node_a, node_b = self.nodes[name_a], self.nodes[name_b]
        else:
            name_a, name_b = a.name, b.name
            node_a, node_b = a, b

        if (name_a, name_b) in self.edges:
            raise ValueError(f'Edge {name_a}({node_a.__class__}) -> {node_b}({node_b.__class__}) already exists.')
        if node_a.output_state != node_b.input_state:
            raise ValueError(f'Mismatched state for {name_a}({node_a.output_state}) -> {node_b}({node_b.input_state}).')
        if node_a.max_outdegree is not None and self.outdegree(node_a) == node_a.max_outdegree:
            raise ValueError(f'Exceeded maximum outdegree of node {name_a}({node_a.__class__}).')
        if node_b.max_indegree is not None and self.indegree(node_b) == node_b.max_indegree:
            raise ValueError(f'Exceeded maximum indegree of node {name_b}({node_b.__class__})..')
        self.edges.append((name_a, name_b))
        self._sub_nec = None
    
    @overload
    def remove_edge(self, node_a: str, node_b: str) -> None: ...
    @overload
    def remove_edge(self, node_a: Node, node_b: Node) -> None: ...
    def remove_edge(self, a: str | Node, b: str | Node) -> None:
        if isinstance(a, str):
            name_a, name_b = a, b
            node_a, node_b = self.nodes[name_a], self.nodes[name_b]
        else:
            name_a, name_b = a.name, b.name
            node_a, node_b = a, b
        
        if (name_a, name_b) not in self.edges:
            raise ValueError(f'Edge {name_a}({node_a.__class__}) -> {node_b}({node_b.__class__}) does not exist.')
        self.edges.remove((name_a, name_b))
        self._sub_nec = None

    @property
    def _nec_nodes(self) -> Dict[str, Node]:
        reachable_from_input = set()
        queue = collections.deque(['input'])
        while queue:
            name = queue.popleft()
            if name not in reachable_from_input:
                reachable_from_input.add(name)
                for child_name in self.children(name):
                    if child_name not in reachable_from_input:
                        queue.append(child_name)
        
        reachable_from_output = set()
        queue = collections.deque(['output'])
        while queue:
            name = queue.popleft()
            if name not in reachable_from_output:
                reachable_from_output.add(name)
                for parent_name in self.parents(name):
                    if parent_name not in reachable_from_output:
                        queue.append(parent_name)

        return {
            name: self.nodes[name]
            for name in reachable_from_input & reachable_from_output
        }
    
    @property
    def _nec_edges(self) -> List[Tuple[str, str]]:
        return [
            edge for edge in self.edges
            if edge[0] in self._nec_nodes and edge[1] in self._nec_nodes
        ]
    
    @staticmethod
    def merge_invalid_nodes(workflow: 'Workflow') -> 'Workflow':
        nodes = {name: node for name, node in workflow.nodes.items()}
        for name, node in nodes.items():
            if node.input_state == node.output_state and \
                (node.max_indegree is None or node.max_indegree > 1):
                if workflow.indegree(name) == 1:
                    parent = workflow.parents(name)[0]
                    workflow.remove_edge(parent, name)
                    for child in workflow.children(name):
                        workflow.remove_edge(name, child)
                        workflow.add_edge(parent, child)
                    workflow.pop_node(name)
        return workflow
    
    @property
    def sub_nec(self) -> 'Workflow':
        if self._sub_nec is None:
            self._sub_nec = self.__class__()
            self._sub_nec.nodes = self._nec_nodes
            self._sub_nec.edges = self._nec_edges
            self._sub_nec = self.merge_invalid_nodes(self._sub_nec)
        return self._sub_nec

    def get_topo_order(self) -> List[str]:
        indegrees = {name: self.indegree(name) for name in self.nodes}
        heap: List[Node] = []
        for name, indegree in indegrees.items():
            if indegree == 0:
                heapq.heappush(heap, (self.nodes[name].to_tuple(), name))
        
        topo_order = []
        while heap:
            _, name = heapq.heappop(heap)
            topo_order.append(name)
            children_name = self.children(name)
            for child_name in children_name:
                indegrees[child_name] -= 1
                if indegrees[child_name] == 0:
                    heapq.heappush(heap, (self.nodes[child_name].to_tuple(), child_name))
        
        if len(topo_order) != len(self.nodes):
            raise RuntimeError('Workflow has cycles.')
        return topo_order
    
    def check(self) -> bool:
        if 'input' not in self.nodes or 'output' not in self.nodes:
            return False
        if len(self.sub_nec.nodes) == 0:
            return False
        try:
            self.sub_nec.get_topo_order()
        except:
            return False
        return True

    def equal(self, other: 'Workflow') -> bool:
        return self.to_tuple() == other.to_tuple()
    
    def copy(self) -> 'Workflow':
        new_workflow = self.__class__()
        new_workflow.nodes = self.nodes.copy()
        new_workflow.edges = deepcopy(self.edges)
        return new_workflow

    async def run(self, messages: Messages) -> Messages:
        if not self.check():
            raise RuntimeError('Invalid workflow.')
        return await self.sub_nec._run(messages)
    
    async def _run(self, messages: Messages) -> Messages:
        topo_order = self.get_topo_order()
        call_results: Dict[str, Messages] = {}
        for name in topo_order:
            node = self.nodes[name]
            if name == 'input':
                input_data = messages.deepcopy()
            else:
                parent_names = [p for p in self.parents(name)]
                parent_outputs = [call_results[parent].deepcopy() for parent in parent_names]
                if node.max_indegree == 1:
                    if len(parent_outputs) != 1:
                        raise ValueError(f"Node {name} expects single input but has {len(parent_outputs)} parents")
                    input_data = parent_outputs[0]
                else:
                    input_data = parent_outputs
            call_results[name] = await node(input_data)
        return call_results['output']
    
    def to_tuple(self) -> tuple:
        topo_order = self.get_topo_order()
        return tuple(
            (
                self.nodes[name].to_tuple(),
                tuple(sorted([topo_order.index(parent) for parent in self.parents(name)])),
                tuple(sorted([topo_order.index(child) for child in self.children(name)]))
            ) for name in topo_order
        )

    def to_dict(self) -> dict:
        return {
            'class_module': self.__class__.__module__,
            'class_name': self.__class__.__name__,
            'nodes': {name: node.to_dict() for name, node in self.nodes.items()},
            'edges': self.edges
        }
    
    @classmethod
    def from_dict(cls, data: dict | str) -> 'Workflow':
        if isinstance(data, str):
            with open(data, 'r', encoding = 'utf-8') as f:
                data = json.load(f)
        module_name = data['class_module']
        class_name = data['class_name']
        module = __import__(module_name, fromlist=[class_name])
        workflow_class = getattr(module, class_name)
        workflow: Workflow = workflow_class()
        workflow.nodes = {name: Node.from_dict(data) for name, data in data['nodes'].items()}
        workflow.edges = data['edges']
        return workflow

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, path: str) -> 'Workflow':
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))

    async def evaluate(self, **kwargs) -> Tuple[float, float, List[Messages]]:
        raise NotImplementedError

class EvaluationWorkflow(Workflow):

    def __init__(self) -> None:
        super().__init__()
        self.add_node('input', EvaluationInput())
        self.add_node('output', EvaluationOutput())

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
        
        return (float(corr), float(pval))

    async def evaluate(
        self,
        dataset: EvaluationDataset,
        max_parallel: int = 8
    ) -> Tuple[float, float, List[Messages]]:
        semaphore = asyncio.Semaphore(max_parallel)
        async def run_with_semaphore(index, inputs: Messages):
            async with semaphore:
                try:
                    messages = await self.run(inputs)
                    return index, messages
                except Exception as e:
                    tqdm.write(f'Data \'{inputs.metadata.id}\' Evaluation Error:\n{e}')
                    return index, None
        
        tasks = []
        for i, messages in enumerate(dataset.inputs):
            task = run_with_semaphore(i, messages)
            tasks.append(task)
        
        messages_predicts: List[Messages] = [None] * len(dataset.inputs)
        for task in tqdm_asyncio.as_completed(tasks, desc = 'EvalWorkflow Evaluation'):
            index, messages = await task
            messages_predicts[index] = messages

        # scores_predicts = []
        # for messages in tqdm(messages_list):
        #     messages = await self.run(messages)
        #     scores_predicts.append(messages.scores)

        n_result = sum([1 if msgs is not None else 0 for msgs in messages_predicts])
        corrs = []
        for eval, scores_labels in dataset.labels.items():
            corr, _ = self.calculate_correlation(
                scores_labels,
                [msgs.scores if msgs is not None else None for msgs in messages_predicts]
            )
            corrs.append(corr)
        max_corr = max(corrs)
        tqdm.write(f'correlations: {corrs}, max correlation: {max_corr}')

        costs = {}
        for messages_predict in messages_predicts:
            if messages_predict is None: continue
            for name, cost in messages_predict.cost.items():
                if name not in costs:
                    costs[name] = 0
                costs[name] += cost
        avg_costs = {name: cost / n_result for name, cost in costs.items()}
        avg_cost = sum(avg_costs.values())
        tqdm.write(f'avg costs: {avg_costs}, avg cost: {avg_cost}')
        return max_corr, avg_cost, messages_predicts

class GenerationWorkflow(Workflow):

    def __init__(self) -> None:
        super().__init__()