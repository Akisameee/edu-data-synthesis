import sys
sys.path.insert(0, '..')

from modules.nodes.base import *
from modules.nodes.utils import *
from modules.datas import EvaluationDataset
from modules.nodes.prompt_templates import *

evaluate_system_template = Template('./modules/nodes/evaluate/evaluate_system.md')
evaluate_user_template = Template('./modules/nodes/evaluate/evaluate_user.md')

class Evaluate(Node):
    input_state = 'assistant'
    output_state = 'scored'
    max_indegree = 1
    
    @retry(max_attempt = 3)
    async def __call__(
        self,
        messages: Messages,
    ) -> Messages:
        system_prompt = evaluate_system_template.format(messages)
        user_prompt = evaluate_user_template.format(messages)

        response, cost = await self.get_response(
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            temperature = 0.0
        )
        messages.cost[self.name] = cost
        scores = extract_json(response.content.strip())
        scores = check_scores(scores, messages.metadata.criteria)

        scores.source = self.llm.model_name
        messages.scores = scores
        return messages
    
class EvaluateICL(Node):
    input_state = 'assistant'
    output_state = 'scored'
    max_indegree = 1

    def __init__(self, llm: Base_LLM, dataset: EvaluationDataset) -> None:
        super().__init__(llm)
        self.dataset = dataset

    def get_human_eval_sample(self, task: str, id: str, count: int = 2):
        
        def get_qid(id: str):
            return '_'.join(id.split('_')[:3])
        
        filtered_samples = [
            s for s in self.dataset
            if get_qid(s['id']) != id and s['task'] == task
        ]

        formatted_samples = []
        for sample in filtered_samples:
            if get_qid(sample['id']) not in [get_qid(s['id']) for s in formatted_samples]:
                formatted_samples.append({
                    'id': sample['id'],
                    'messages': Messages([Message(**message) for message in sample['message']]),
                    'scores': EvalScores([EvalScore(**score) for score in sample['scores']])
                })

        formatted_samples.sort(key = lambda s: s['scores'].sum())
        # points = [0.333, 0.666]
        # points = [0, 0.333]
        # points = [0.666, 1]
        points = [0, 1]

        def get_sample_point(samples, ratio):
            return samples[round((len(samples) - 1) * ratio)]

        text_samples = ''
        for idx, point in enumerate(points):
            sample = get_sample_point(formatted_samples, point)
            text_samples += f"样例{idx + 1}对话：{sample['messages'].to_json()}\n"
            text_samples += f"样例{idx + 1}评估结果：{sample['scores'].to_json()}\n"
        text_samples += '\n'

        return text_samples

    @retry(max_attempt = 3)
    async def __call__(
        self,
        messages: Messages
    ) -> Messages:

        prompt = evaluation_cl_template.format(
            scenario = messages.scores['scenario'],
            message = messages.to_list(),
            criteria = messages.scores['criteria'],
            samples = self.get_human_eval_sample(
                messages.scores['task'],
                messages.scores['id']
            )
        )

        response, cost = await self.get_response(
            messages = [{'role': 'user', 'content': prompt}, ],
            temperature = 0.0
        )
        messages.cost[self.name] = cost
        scores = extract_json(response.content.strip())
        scores = check_scores(scores, messages.scores['criteria'])

        scores.source = self.llm.model_name
        messages.scores = scores
        return messages