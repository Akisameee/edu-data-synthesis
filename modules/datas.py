from modules.utils import *
from modules.base import *

SCENARIO_DIR = './data/scenario'
CRITERIA_DIR = './data/criteria'

def read_scenarios(theme_dir: str, language: str) -> Dict[str, Scenario]:

    with open(os.path.join(theme_dir, f'scenarios_{language}.json'), 'r', encoding = 'utf-8') as file:
        scenarios = json.load(file)
    return {
        name: Scenario(**scenario)
        for name, scenario in scenarios.items()
    }

def read_criteria(metrics_dir: str, language: str, return_dict: bool = False) -> Criteria | Dict[str, Any]:
    with open(os.path.join(metrics_dir, f'metrics_{language}_whiten.json'), 'r', encoding = 'utf-8') as file:
        eval_metrics = json.load(file)
    if return_dict:
        return eval_metrics
    else:
        return Criteria(list(eval_metrics.values()))

def read_criteria_map(metrics_dir: str, language: str) -> Tuple[Criteria, Dict[str, Criteria]]:
    with open(os.path.join(metrics_dir, f'metrics_{language}_whiten.json'), 'r', encoding = 'utf-8') as file:
        eval_metrics = json.load(file)
    with open(os.path.join(metrics_dir, 'metrics_map.json'), 'r', encoding = 'utf-8') as file:
        metrics_map = json.load(file)
    criteria_map = {
        scenario: Criteria([eval_metrics[idx] for idx in criteria_idxs])
        for scenario, criteria_idxs in metrics_map.items()
    }
    return criteria_map

class Dataset():
    name: str
    language: Literal['zh', 'en']
    scenarios: Dict[str, Scenario]
    criteria: Criteria
    criteria_map: Dict[str, Criteria]
    
    def __init__(self, language: Literal['zh', 'en'] = 'en') -> None:
        self.language = language
        self.scenarios = read_scenarios(SCENARIO_DIR, language)
        self.criteria = read_criteria(CRITERIA_DIR, language)
        self.criteria_map = read_criteria_map(CRITERIA_DIR, language)

        criteria_dict = read_criteria(CRITERIA_DIR, language, True)
        criteria_dict_other = read_criteria(
            CRITERIA_DIR, 'zh' if language == 'en' else 'en', True
        )
        self.name_map = {
            criteria_dict_other[key]['name']: criterion['name']
            for key, criterion in criteria_dict.items()
        }

class EvaluationDataset(Dataset):
    inputs: List[Messages]
    labels: Dict[str, List[EvalScores]]

    def __init__(
        self,
        eval_path: str = None,
        language: Literal['zh', 'en'] = 'en'
    ) -> None:
        super().__init__(language)

        if eval_path is None:
            self.inputs, self.labels = [], {}
            return

        if 'train' in eval_path:
            self.name = 'eval_train'
        elif 'val' in eval_path:
            self.name = 'eval_val'
        else:
            raise ValueError(f'Invalid eval path: {eval_path}.')
        self.eval_datas = read_jsonl(eval_path)

        human_eval_datas = {}
        evals = []
        for eval_data in self.eval_datas:
            # if eval_data['language'] == 'zh': continue
            if not eval_data['eval'].startswith('human_'):
                continue
            if eval_data['id'] not in human_eval_datas:
                messages = Messages(eval_data['message'])
                messages.source = eval_data['gen']
                messages.metadata = MetaData(
                    id = eval_data['id'],
                    language = eval_data['language'],
                    task = eval_data['task'],
                    scenario = self.scenarios[eval_data['task']],
                    criteria = self.criteria_map[eval_data['task']],
                )
                human_eval_datas[eval_data['id']] = {'messages': messages}
            if eval_data['eval'] not in evals:
                evals.append(eval_data['eval'])
            scores = EvalScores(eval_data['scores'])
            scores.source = eval_data['eval']
            for score in scores:
                if score.criterion not in self.criteria.names:
                    score.criterion = self.name_map[score.criterion]
            human_eval_datas[eval_data['id']][eval_data['eval']] = scores

        self.inputs = []
        self.labels = {eval: [] for eval in evals}
        for id, data in human_eval_datas.items():
            self.inputs.append(data['messages'])
            for eval in evals:
                self.labels[eval].append(data[eval])

    def __len__(self) -> int:
        return len(self.inputs)
    
    def __getitem__(self, idx: int) -> Tuple[Messages, Dict[str, EvalScores]]:
        return self.inputs[idx], {
            eval_name: scores_list[idx]
            for eval_name, scores_list in self.labels.items()
        }
    
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
    
    def sub_criterion(self, criterion_name: str) -> 'EvaluationDataset':
        sub_dataset = self.__class__(language = self.language)
        name_prefix = criterion_name.lower().replace(' ', '_')
        sub_dataset.name = f'{name_prefix}_{self.name}'
        sub_dataset.labels = {eval: [] for eval in self.labels.keys()}
        for messages, scores_dict in self:
            criterion = messages.metadata.criteria[criterion_name]
            if criterion is not None:
                input = messages.deepcopy()
                input.metadata.criteria = Criteria([criterion])
                sub_dataset.inputs.append(input)
                for eval_name, scores in scores_dict.items():
                    single_scores = EvalScores([scores[criterion_name]])
                    single_scores.source = scores.source
                    sub_dataset.labels[eval_name].append(single_scores)
        return sub_dataset
    
    def get_task(self, task: str) -> 'EvaluationDataset':
        pass
