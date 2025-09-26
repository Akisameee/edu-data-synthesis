import sys

from modules.tools import List
sys.path.insert(0, '..')

from modules.base import List
from modules.models import Base_LLM
from modules.nodes.base import *
from modules.nodes.utils import *
from modules.datas import EvaluationDataset
from modules.nodes.prompt_templates import *

evaluate_system_template = Template('./modules/nodes/evaluate/evaluate_system.md')
evaluate_user_template = Template('./modules/nodes/evaluate/evaluate_user.md')

SAMPLE_PATH = './data/eval_data/eval_samples.jsonl'

class Evaluate(Node):
    input_state = 'assistant'
    output_state = 'scored'
    max_indegree = 1

    fewshot_samples: List[Messages] = []

    def __init__(
        self,
        llm: Optional[str | Base_LLM],
        tools: Optional[List[str]] = [],
        fewshot_metadatas: List[Dict[str, str]] = [],
        cache: bool = False,
        max_cache_count: int = 5
    ) -> None:
        super().__init__(llm, tools, cache, max_cache_count)
        self.fewshot_samples = self.get_fewshot_samples(fewshot_metadatas)

    @staticmethod
    def get_fewshot_samples(
        fewshot_metadatas: List[Dict[str, str]]
    ) -> List[Messages]:
        if len(fewshot_metadatas) == 0: return []
        fewshot_samples = []
        dataset = EvaluationDataset(SAMPLE_PATH)
        samples_dict = {}
        for sample in fewshot_metadatas:
            if sample['id'] not in samples_dict:
                samples_dict[sample['id']] = []
            samples_dict[sample['id']].append(sample['eval'])
        for messages, scores_dict in dataset:
            if messages.metadata.id in samples_dict:
                evals = samples_dict[messages.metadata.id]
                for eval, scores in scores_dict.items():
                    if eval in evals:
                        messages = messages.deepcopy()
                        messages.scores = scores
                        fewshot_samples.append(messages)
        return fewshot_samples

    @staticmethod
    def get_fewshot_messages(
        fewshot_samples: List[Messages],
        criteria_names: List[str]
    ) -> List[Dict[str, str]]:
        fewshot_messages = []
        for sample in fewshot_samples:
            user_fewshot_prompt = evaluate_user_template.format(
                sample, fewshot = True
            )
            sub_scores = EvalScores([sample.scores.get_score(c_n) for c_n in criteria_names])
            fewshot_messages += [
                {'role': 'user', 'content': user_fewshot_prompt},
                {'role': 'assistant', 'content': json.dumps(
                    sub_scores.to_list(), ensure_ascii = False
                )}
            ]
        return fewshot_messages
    
    @retry(max_attempt = 3)
    async def run(
        self,
        messages: Messages,
    ) -> Messages:
        if messages.metadata.id and messages.metadata.id in [s.metadata.id for s in self.fewshot_samples]:
            raise RuntimeError(f'Data {messages.metadata.id} has been used as a fewshot sample.')
        
        system_prompt = evaluate_system_template.format(
            messages, tools = [tool.name for tool in self.tools]
        )
        user_prompt = evaluate_user_template.format(
            messages, fewshot = False
        )

        # with open('test_system_prompt.md', 'w', encoding='utf-8') as f:
        #     f.write(system_prompt)
        # with open('test_user_prompt.md', 'w', encoding='utf-8') as f:
        #     f.write(user_prompt)

        response, cost = await self.get_response(
            messages = [
                {'role': 'system', 'content': system_prompt},
                *self.get_fewshot_messages(
                    self.fewshot_samples,
                    messages.metadata.criteria.names
                ),
                {'role': 'user', 'content': user_prompt},
            ],
            temperature = 0.0
        )
        messages.cost[self.name] = cost
        scores = extract_json(response.content.strip())
        scores = check_scores(scores, messages.metadata.criteria)

        scores.source = self.llm.name
        messages.scores = scores
        return messages
    
    def to_tuple(self) -> tuple:
        return (
            *super().to_tuple(),
            sorted([
                {'id': sample.metadata.id, 'eval': sample.scores.source}
                for sample in self.fewshot_samples
            ])
        )

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            'fewshot_metadatas': sorted([
                {'id': sample.metadata.id, 'eval': sample.scores.source}
                for sample in self.fewshot_samples
            ])
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Node':
        node: 'Evaluate' = super().from_dict(data)
        node.fewshot_samples = Evaluate.get_fewshot_samples(data.get('fewshot_metadatas', []))
        return node