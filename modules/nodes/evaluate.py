import sys
sys.path.insert(0, '..')

from modules.nodes.base import *
from modules.nodes.prompt_templates import *

class Evaluate(Node):
    input_state = 'assistant'
    output_state = 'scored'
    max_indegree = 1

    @staticmethod
    def check_scores(scores: List[Dict[str, float | str]], criteria: Criteria):

        extra_criteria = []
        for score in scores:
            criterion = [c.name for c in criteria if score['criterion'] in c.name]
            if len(criterion) > 1:
                invalid_criteria = score['criterion']
                raise ValueError(f'[Score Parse Error] Invalid criteria: {invalid_criteria}.')
            if len(criterion) == 0:
                extra_criteria.append(score['criterion'])
                continue
            score['criterion'] = criterion[0]

            value = score['score']
            if not isinstance(value, (int, float)):
                raise ValueError(f'[Score Parse Error] Invalid score value: {value}.')
        
        scores = [score for score in scores if score['criterion'] not in extra_criteria]
            
        if set(score['criterion'] for score in scores) != \
            set(c.name for c in criteria):
            invalid_criteria = [score['criterion'] for score in scores]
            required_criteria = [c.name for c in criteria]
            raise ValueError(f'[Score Parse Error] Invalid criteria: {invalid_criteria}, required: {required_criteria}.')
        
        return scores
    
    @retry(max_attempt = 3)
    async def __call__(
        self,
        messages: Messages,
    ) -> Messages:
        prompt = evaluation_template.format(
            scenario = messages.metadata.scenario.__dict__,
            message = messages.to_json(),
            criteria = messages.metadata.criteria.to_json()
        )

        completion = await self.llm.get_response(
            messages = [{'role': 'user', 'content': prompt}, ],
            temperature = 0.0
        )
        messages.cost[self.name] = self.llm.cost(completion)
        response = completion.choices[0].message.content.strip()

        scores = extract_json(response)
        self.check_scores(scores, messages.metadata.criteria)
        scores = EvalScores([EvalScore(**score) for score in scores])
        scores.source = self.llm.model_name
        messages.scores = scores
        return messages
    
class EvaluateICL(Node):
    input_state = 'assistant'
    output_state = 'scored'
    max_indegree = 1

    def __init__(self, llm: Base_LLM = None) -> None:
        super().__init__(llm)
        eval_samples = read_jsonl('./eval_res/eval_samples.jsonl')
        sub_eval_samples = read_jsonl('./eval_res/sub_eval_samples.jsonl')
        sub_eval_sample_ids = [s['id'] for s in sub_eval_samples]
        self.samples = [s for s in eval_samples if s['id'] not in sub_eval_sample_ids]

    def get_human_eval_sample(self, task: str, id: str, count: int = 2):
        
        def get_qid(id: str):
            return '_'.join(id.split('_')[:3])
        
        filtered_samples = [
            s for s in self.samples
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
        Evaluate.check_scores(scores, messages.scores['criteria'])
        scores = EvalScores([EvalScore(**score) for score in scores])
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

        Evaluate.check_scores(scores, messages.metadata.criteria)
        scores = EvalScores([EvalScore(**score) for score in scores])
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

        Evaluate.check_scores(scores, messages.metadata.criteria)
        scores = EvalScores([EvalScore(**score) for score in scores])
        scores.source = self.llm.model_name
        messages.scores = scores
        return messages