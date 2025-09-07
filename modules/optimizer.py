import json
import datetime
from tqdm import tqdm
import asyncio
import os

from modules.logging import TqdmLogger
from modules.models import get_model
from modules.workflow import *
from modules.utils import *

RES_DIR = './opt_res'

class Optimizer:

    def __init__(self, init_workflow: Workflow) -> None:
        self.workflow: Workflow = init_workflow
        self.workflows: Dict[Workflow, float] = {}
        self.logger = TqdmLogger(f'{self.__class__.__name__}_Opt', RES_DIR)

    def add_node(self, node: Node) -> None:
        idx = 0
        while True:
            name = f'{node.__class__.__name__}_{idx}'
            if name not in self.workflow.nodes:
                break
            else: idx += 1
        self.workflow.add_node(name, node)

    # def 

    def run(self, **kwargs) -> Workflow:
        raise NotImplementedError
    
class LocalSearch(Optimizer):

    @dataclass
    class Operation:
        func: callable
        args: tuple = None
        kwargs: dict = None

    def get_mutation_ops(self, workflow: Workflow, mutated_edges: Tuple[str, str] = []) -> List[Operation]:
        ops: List[self.Operation] = []
        for parent in workflow.nodes.values():
            for child in workflow.nodes.values():
                edge = (parent.name, child.name)
                if child == parent or edge in mutated_edges:
                    continue
                if edge not in workflow.edges:
                    if parent.output_state == child.input_state and \
                        (parent.max_outdegree is None or workflow.outdegree(parent) < parent.max_outdegree) and \
                        (child.max_indegree is None or workflow.indegree(child) < child.max_indegree):
                        ops.append(self.Operation('add_edge', edge))
                else:
                    ops.append(self.Operation('remove_edge', edge))
        return ops
    
    def get_neighbor(self, workflow: Workflow, max_mutation_ops: int) -> List[Workflow]:
        neighbors: Dict[List[self.Operation], Workflow] = {}
        mutated_edges: Tuple[str, str] = []
        for _ in max_mutation_ops:
            pass
    
    def run(
        self,
        max_mutation_ops: int = 3,
        max_iter: int = 10
    ):
        pass