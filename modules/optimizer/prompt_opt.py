import json
import random
from tqdm import tqdm
import asyncio
import os
import optuna
from typing import Callable

from modules.optimizer.base import *

RES_DIR = get_config_value('prompt_opt_dir')

class PromptOptimizer(Optimizer):

    def __init__(
        self,
        init_node: Node,
        train_dataset: Dataset
    ) -> None:
        super().__init__(train_dataset, RES_DIR)
        self.node: Node = init_node
        self.nodes_evaluated = self.load_scores()

    def check_evaluated(self, node: Node) -> Optional[float]:
        for node_evaluated in self.nodes_evaluated:
            if str(node.to_tuple()) == node_evaluated['tuple_tag']:
                return node_evaluated['score']
        return None

    def evaluate(self, node: Node, n_eval: int = 1) -> float:
        eval_res = self.check_evaluated(node)
        if eval_res is None:
            eval_workflow = EvaluationWorkflow()
            eval_workflow.add_node('base_node', node)
            eval_workflow.add_edge('input', 'base_node')
            eval_workflow.add_edge('base_node', 'output')
            scores = []
            for _ in range(n_eval):
                score, _, _ = asyncio.run(eval_workflow.evaluate(self.train_dataset))
                scores.append(score)
            score_avg = sum(scores) / n_eval
            self.nodes_evaluated.append({
                'node': node,
                'tuple_tag': str(node.to_tuple()),
                'score': score_avg
            })
            self.save_scores(self.nodes_evaluated)
        else:
            score_avg = eval_res
        self.logger.info(f'Evaluation result for {node.to_dict()}:\nscore: {score_avg}.')
        return score_avg 
    
    def load_scores(self) -> List[Dict[str, Node | float]]:
        if not os.path.exists(self.scores_path):
            return []
        datas: List[Dict[str, str | float]] = read_jsonl(self.scores_path)
        nodes_evaluated: List[Dict[str, Workflow | float]] = []
        for data in datas:
            data['node'] = Node.from_dict(data['node'])
            nodes_evaluated.append(data)
        return nodes_evaluated

    def save_scores(self, nodes_evaluated: List[Dict[str, Node | float]]) -> None:
        datas = []
        for node_evaluated in nodes_evaluated:
            node_evaluated['node'] = node_evaluated['node'].to_dict()
            datas.append(node_evaluated)
        write_jsonl(self.scores_path, datas)

    def run(self, **kwargs) -> Node:
        raise NotImplementedError
    
class FewshotSampleOptimizer(PromptOptimizer):

    def set_fewshot_samples(
        self,
        node: Node,
        indices: List[int]
    ) -> Node:
        pass

    def run(
        self,
        init_node: Node,
        max_iter: int = 10,
        max_samples: int = 2
    ) -> Node:
        
        def objective(trial: optuna.Trial):
            num_samples = trial.suggest_int('num_samples', 0, max_samples)
            if num_samples == 0:
                return self.evaluate([])
            selected_indices = []
            for i in range(num_samples):
                available_indices = [
                    idx for idx in range(self.total_samples) 
                    if idx not in selected_indices
                ]
                selected_idx = trial.suggest_categorical(
                    f'selected_sample_{i}', available_indices
                )
                selected_indices.append(selected_idx)
            return self.evaluate(selected_indices)
        
        study = self.optimize(objective, max_iter)
        best_indices = self.get_best_selection(study)
        return self.set_fewshot_samples(init_node, best_indices)
    
    def optimize(
        self,
        obj_func: Callable,
        n_trials: int = 10,
        show_progress: bool = True
    ) -> optuna.Study:
        study = optuna.create_study(direction = 'maximize')
        study.optimize(
            obj_func,
            n_trials = n_trials, 
            show_progress_bar = show_progress
        )
        return study
    
    def get_best_selection(self, study: optuna.Study) -> List[int]:
        if study.best_params is None:
            return None
        num_samples = study.best_params.get('num_samples', 0)
        selected_indices = []
        for i in range(num_samples):
            key = f'selected_sample_{i}'
            if key in study.best_params:
                selected_indices.append(study.best_params[key])
        return selected_indices