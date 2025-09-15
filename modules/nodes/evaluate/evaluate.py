import sys
sys.path.insert(0, '..')

from modules.nodes.base import *
from modules.nodes.utils import *
from modules.datas import EvaluationDataset
from modules.nodes.prompt_templates import *

evaluate_sys_template = Template('./modules/nodes/evaluate/evaluate_system.md')
evaluate_user_template = Template('./modules/nodes/evaluate/evaluate_user.md')
evaluate_single_sys_template = Template('./modules/nodes/evaluate/single_system.md')
evaluate_single_user_template = Template('./modules/nodes/evaluate/single_user.md')

class Evaluate(Node):
    input_state = 'assistant'
    output_state = 'scored'
    max_indegree = 1
    
    @retry(max_attempt = 3)
    async def __call__(
        self,
        messages: Messages,
    ) -> Messages:
        if len(messages.metadata.criteria) > 1:
            sys_prompt = evaluate_sys_template.format(messages)
            user_prompt = evaluate_user_template.format(messages)
        elif len(messages.metadata.criteria) == 1:
            sys_prompt = evaluate_single_sys_template.format(messages)
            user_prompt = evaluate_single_user_template.format(
                messages, criterion = messages.metadata.criteria[0].to_md(1)
            )
        else:
            raise ValueError('Invalid criteria.')
        # with open('test_prompt.md', 'w', encoding='utf-8') as f:
        #     f.write(user_prompt)

        completion = await self.llm.get_response(
            messages = [
                {'role': 'system', 'content': sys_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            temperature = 0.0
        )
        messages.cost[self.name] = self.llm.cost(completion)
        response = completion.choices[0].message.content.strip()

        scores = extract_json(response)
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
            message = messages.to_json(),
            criteria = messages.scores['criteria'],
            samples = self.get_human_eval_sample(
                messages.scores['task'],
                messages.scores['id']
            )
        )

        completion = await self.llm.get_response(
            messages = [{'role': 'user', 'content': prompt}, ],
            temperature = 0.0
        )
        messages.cost[self.name] = self.llm.cost(completion)
        response = completion.choices[0].message.content.strip()

        scores = extract_json(response)
        scores = check_scores(scores, messages.scores['criteria'])

        scores.source = self.llm.model_name
        messages.scores = scores
        return messages

class EvaluateSingle(Node):
    input_state = 'assistant'
    output_state = 'scored'
    max_indegree = 1

    @retry(max_attempt = 3)
    async def __call__(
        self,
        messages: Messages
    ) -> Messages:

        if isinstance(self.llm, LLM_API):
            return await self._call_llm_api(messages)
        elif isinstance(self.llm, RM_HF):
            return self._call_rm_hf(messages)
    
    async def _call_llm_api(
        self,
        messages: Messages
    ) -> Messages:
        scores = []
        for criterion in messages.metadata.criteria:
            prompt = evaluation_single_template.format(
                scenario = messages.metadata.scenario.__dict__,
                message = messages.to_json(),
                criterion = criterion.__dict__
            )

            completion = await self.llm.get_response(
                messages = [{'role': 'user', 'content': prompt}, ],
                temperature = 0.0
            )
            messages.cost[self.name] = self.llm.cost(completion)
            response = completion.choices[0].message.content.strip()
            scores.append(extract_json(response))

        scores = check_scores(scores, messages.metadata.criteria)

        scores.source = self.llm.model_name
        messages.scores = scores
        return messages
    
    def _call_rm_hf(
        self,
        messages: Messages
    ) -> Messages:
        scores = []
        for criterion in messages.metadata.criteria:
            prompt = evaluation_single_template.format(
                scenario = messages.metadata.scenario.__dict__,
                message = messages.to_json(),
                criterion = criterion.__dict__
            )

            rewards = []
            for score in range(10):
                response_template = '```json{{"criterion": "{criterion_name}", "score": {score}, "reason": "{reason}"}}```'
                response = response_template.format(
                    criterion_name = criterion['metric'],
                    score = str(score + 1),
                    reason = criterion['levels'][-(int(score / 2) + 1)]
                )
                reward = self.llm.get_reward(
                    messages = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': response}]
                )
                rewards.append({
                    'reward': reward,
                    'response': extract_json(response)
                })
            rewards.sort(key = lambda r: r['reward'], reverse = True)
            scores.append(rewards[0]['response'])

        scores = check_scores(scores, messages.metadata.criteria)

        scores.source = self.llm.model_name
        messages.scores = scores
        return messages