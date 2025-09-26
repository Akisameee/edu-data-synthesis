import json
import random
from tqdm import tqdm
import asyncio
import os

from modules.optimizer.base import *

RES_DIR = get_config_value('workflow_opt_dir')

class WorkflowOptimizer(Optimizer):

    def __init__(
        self,
        init_workflow: Workflow,
        train_dataset: Dataset,
        cost_weight: float = -1
    ) -> None:
        super().__init__(train_dataset, RES_DIR)
        self.workflow: Workflow = init_workflow
        self.cost_weight = cost_weight
        
        self.workflows_evaluated = self.load_scores()

    def check_evaluated(self, workflow: Workflow) -> Optional[Tuple[float, float]]:
        for workflow_evaluated in self.workflows_evaluated:
            if str(workflow.sub_nec.to_tuple()) == workflow_evaluated['tuple_tag']:
                return workflow_evaluated['score'], workflow_evaluated['cost']
        return None

    def evaluate(self, workflow: Workflow, n_eval: int = 1) -> float:
        eval_res = self.check_evaluated(workflow)
        if eval_res is None:
            scores, costs = [], []
            for _ in range(n_eval):
                score, cost, _ = asyncio.run(workflow.evaluate(self.train_dataset))
                scores.append(score)
                costs.append(cost)
            score_avg = sum(scores) / n_eval
            cost_avg = sum(costs) / n_eval
            self.workflows_evaluated.append({
                'workflow': workflow.sub_nec,
                'tuple_tag': str(workflow.sub_nec.to_tuple()),
                'score': score_avg,
                'cost': cost_avg
            })
            self.save_scores(self.workflows_evaluated)
        else:
            score_avg, cost_avg = eval_res
        self.logger.info(f'Evaluation result for {workflow.edges}:\nscore: {score_avg}, cost: {cost_avg}.')
        return score_avg + cost_avg * self.cost_weight
    
    def load_scores(self) -> List[Dict[str, Workflow | float]]:
        if not os.path.exists(self.scores_path):
            return []
        datas: List[Dict[str, str | float]] = read_jsonl(self.scores_path)
        # with open(self.scores_path, 'r') as f:
        #     datas: List[Dict[str, str | float]] = json.load(f)
        workflows_evaluated: List[Dict[str, Workflow | float]] = []
        for data in datas:
            data['workflow'] = Workflow.from_dict(data['workflow'])
            workflows_evaluated.append(data)
        return workflows_evaluated

    def save_scores(self, workflows_evaluated: List[Dict[str, Workflow | float]]) -> None:
        datas = []
        for workflow_evaluated in workflows_evaluated:
            workflow_evaluated['workflow'] = workflow_evaluated['workflow'].to_dict()
            datas.append(workflow_evaluated)
        write_jsonl(self.scores_path, datas)
        # with open(self.scores_path, 'w') as f:
        #     json.dump(datas, f, indent=4)

    def run(self, **kwargs) -> Workflow:
        raise NotImplementedError
    
class LocalSearch(WorkflowOptimizer):

    @dataclass
    class Operation:
        func: str
        args: tuple = None
        kwargs: dict = None

        def apply(self, workflow: Workflow) -> Workflow:
            workflow = workflow.copy()
            args = self.args if self.args is not None else ()
            kwargs = self.kwargs if self.kwargs is not None else {}
            getattr(workflow, self.func)(*args, **kwargs)
            return workflow

    def get_mutation_ops(
        self, workflow: Workflow,
        mutated_edges: List[Tuple[str, str]] = []
    ) -> List[Operation]:
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
    
    def get_neighbor(
        self,
        workflow: Workflow,
        max_mutation_ops: int,
        mutated_edges: List[Tuple[str, str]] = []
    ) -> Workflow:
        mutated_workflows: List[Tuple[Workflow, Tuple[str, str]]] = []
        mutation_ops = self.get_mutation_ops(workflow, mutated_edges)
        random.shuffle(mutation_ops)

        for op in mutation_ops:
            mutated_workflow = op.apply(workflow)
            if mutated_workflow.check() and self.check_evaluated(mutated_workflow) is None:
                return mutated_workflow
            mutated_workflows.append((mutated_workflow, op.args))
        
        if max_mutation_ops > 0:
            for mutated_workflow, mutated_edge in mutated_workflows:
                neighbor = self.get_neighbor(mutated_workflow, max_mutation_ops - 1, mutated_edges + [mutated_edge])
                if neighbor is not None:
                    return neighbor
        return None
    
    def run(
        self,
        max_iter: int = 10,
        max_mutation_ops: int = 4
    ) -> Workflow:
        self.logger.info('Evaluating initial workflow...')
        self.evaluate(self.workflow)

        for iteration in range(max_iter):
            self.logger.info(f'Optimization step: {iteration + 1}/{max_iter}:')
            neighbor = self.get_neighbor(self.workflow, max_mutation_ops)
            if neighbor is None:
                self.logger.info(f'Unable to find avaliable neighbors.')
                break
            self.logger.info(f'Found neighbor: {neighbor.sub_nec.edges}.')

            if self.evaluate(neighbor) > self.evaluate(self.workflow):
                self.workflow = neighbor
        
        return self.workflow
    