from modules.models import get_model
from modules.utils import *
from modules.base import *

class Dataset():
    pass

class EvaluationDataset(Dataset):
    inputs: List[Messages]
    labels: Dict[str, List[EvalScores]]

    def __init__(self, eval_path: str, language: str = 'zh') -> None:
        super().__init__()
        self.scenarios = read_scenarios('./data/scenario', language)
        self.criterias = read_criterias('./data/criteria', language)  
        self.eval_datas = read_jsonl(eval_path)

class EvaluationFullCriteria(EvaluationDataset):

    def __init__(self, eval_path: str, language: str = 'zh') -> None:
        super().__init__(eval_path, language)

        human_eval_datas = {}
        evals = []
        for eval_data in self.eval_datas:

            if not eval_data['eval'].startswith('human_'):
                continue
            
            if eval_data['id'] not in human_eval_datas:
                scenario = self.scenarios[eval_data['task']]
                messages = Messages(eval_data['message'])
                messages.source = eval_data['gen']
                messages.meta_data = {
                    'id': eval_data['id'],
                    'task': eval_data['task'],
                    'scenario': scenario,
                    'criteria': self.criterias[scenario['task']],
                }
                human_eval_datas[eval_data['id']] = {'messages': messages}
            if eval_data['eval'] not in evals:
                evals.append(eval_data['eval'])
            human_eval_datas[eval_data['id']][eval_data['eval']] = EvalScores(eval_data['scores'])

        self.inputs = []
        self.labels = {eval: [] for eval in evals}
        for id, data in human_eval_datas.items():
            self.inputs.append(data['messages'])
            for eval in evals:
                self.labels[eval].append(data[eval])


class EvaluationSingleCriterion(EvaluationDataset):
    inputs: List[Messages]
    labels: Dict[str, List[EvalScores]]

    def __init__(self, eval_path: str, language: str = 'zh') -> None:
        super().__init__(eval_path, language)

        human_eval_datas = {}
        evals = []
        for eval_data in self.eval_datas:

            if not eval_data['eval'].startswith('human_'):
                continue
            
            if eval_data['id'] not in human_eval_datas:
                scenario = self.scenarios[eval_data['task']]
                messages = Messages(eval_data['message'])
                messages.source = eval_data['gen']
                messages.meta_data = {
                    'id': eval_data['id'],
                    'task': eval_data['task'],
                    'scenario': scenario,
                    'criteria': self.criterias[scenario['task']],
                }
                human_eval_datas[eval_data['id']] = {'messages': messages}
            if eval_data['eval'] not in evals:
                evals.append(eval_data['eval'])
            human_eval_datas[eval_data['id']][eval_data['eval']] = EvalScores(eval_data['scores'])

        self.inputs = []
        self.labels = {eval: [] for eval in evals}
        for id, data in human_eval_datas.items():
            self.inputs.append(data['messages'])
            for eval in evals:
                self.labels[eval].append(data[eval])