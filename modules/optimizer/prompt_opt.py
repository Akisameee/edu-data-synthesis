import json
import random
from tqdm import tqdm
import asyncio
import os
import optuna

from modules.optimizer.base import *
from modules.optimizer.base import Dataset, Node

RES_DIR = './prompt_opt_res'

class PromptOptimizer(Optimizer):

    def __init__(
        self,
        init_node: Node,
        train_dataset: Dataset,
        cost_weight: float = -1
    ) -> None:
        super().__init__(train_dataset, RES_DIR)
        self.node: Node = init_node
        self.cost_weight = cost_weight

        self.nodes_evaluated = self.load_scores()

    def check_evaluated(self, node: Node) -> Optional[Tuple[float, float]]:
        for node_evaluated in self.nodes_evaluated:
            if str(node.to_tuple()) == node_evaluated['tuple_tag']:
                return node_evaluated['score'], node_evaluated['cost']
        return None

    def evaluate(self, node: Node, n_eval: int = 1) -> float:
        eval_workflow = EvaluationWorkflow()
        eval_workflow.add_node('base_node', node)
        eval_workflow.add_edge('input', 'base_node')
        eval_workflow.add_edge('base_node', 'output')
    
    def load_scores(self) -> List[Dict[str, Node | float]]:
        pass

    def save_scores(self, nodes_evaluated: List[Dict[str, Node | float]]) -> None:
        pass

    def run(self, **kwargs) -> Node:
        raise NotImplementedError
    
class FewshotSampleOptimizer(PromptOptimizer):

    def run(
        self,
        max_iter: int = 10,
        max_samples: int = 2
    ) -> Node:
        pass