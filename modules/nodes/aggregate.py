import numpy as np

import sys
sys.path.insert(0, '..')

from modules.base import Base_LLM
from modules.models import Base_LLM
from modules.nodes.base import *
from modules.nodes.prompt_templates import *
from modules.nodes.evaluate import Evaluate

class EvaluationAverage(Node):
    input_state = 'scored'
    output_state = 'scored'
    max_indegree = None

    def __init__(self) -> None:
        super().__init__(None)

    @retry(max_attempt = 3)
    async def __call__(self, messages_list: List[Messages]) -> Messages:
        if len(messages_list) == 1:
            return messages_list[0]
        
        messages = messages_list[0]
        scores_avg = []
        for criterion in messages.scores.criteria:
            scores_avg.append(EvalScore(
                criterion = criterion,
                score = sum([
                    msgs.scores.get_score(criterion).score for msgs in messages_list
                ]) / len(messages_list),
                reason = '\n'.join([
                    msgs.scores.get_score(criterion).reason for msgs in messages_list
                ])
            ))
        messages.scores = EvalScores(scores_avg)
        return messages
    
class EvaluationMax(Node):
    input_state = 'scored'
    output_state = 'scored'
    max_indegree = None

    def __init__(self) -> None:
        super().__init__(None)

    @retry(max_attempt = 3)
    async def __call__(self, messages_list: List[Messages]) -> Messages:
        if len(messages_list) == 1:
            return messages_list[0]
        
        messages = messages_list[0]
        scores_max = []
        for criterion in messages.scores.criteria:
            max_idx = np.argmax([msgs.scores.get_score(criterion).score for msgs in messages_list])
            scores_max.append(EvalScore(
                criterion = criterion,
                score = messages_list[max_idx].scores.get_score(criterion).score,
                reason = messages_list[max_idx].scores.get_score(criterion).reason
            ))
        messages.scores = EvalScores(scores_max)
        return messages

class EvaluationMin(Node):
    input_state = 'scored'
    output_state = 'scored'
    max_indegree = None

    def __init__(self) -> None:
        super().__init__(None)

    @retry(max_attempt = 3)
    async def __call__(self, messages_list: List[Messages]) -> Messages:
        if len(messages_list) == 1:
            return messages_list[0]
        
        messages = messages_list[0]
        scores_min = []
        for criterion in messages.scores.criteria:
            max_idx = np.argmin([msgs.scores.get_score(criterion).score for msgs in messages_list])
            scores_min.append(EvalScore(
                criterion = criterion,
                score = messages_list[max_idx].scores.get_score(criterion).score,
                reason = messages_list[max_idx].scores.get_score(criterion).reason
            ))
        messages.scores = EvalScores(scores_min)
        return messages

class EvaluationAggregation(Node):
    input_state = 'scored'
    output_state = 'scored'
    max_indegree = None

    @retry(max_attempt = 3)
    async def __call__(self, messages_list: List[Messages]) -> Messages:
        if len(messages_list) == 1:
            return messages_list[0]

        messages = messages_list[0].copy()
        prompt = evaluation_aggregate_template.format(
            scenario = messages.meta_data['scenario'],
            message = messages.to_json(),
            criteria = messages.meta_data['criteria']
        ) + '\n' + ''.join([
            f'Scores {idx}:\n{msgs.scores.to_json()}\n'
            for idx, msgs in enumerate(messages_list)
        ])
        
        completion = await self.llm.get_response(
            messages = [{'role': 'user', 'content': prompt}, ]
        )
        response = completion.choices[0].message.content.strip()

        scores = extract_json(response)
        Evaluate.check_scores(scores, messages.meta_data['criteria'])
        scores = EvalScores(scores)
        scores.source = self.llm.model_name
        messages.scores = scores
        messages.cost[self.name] = self.llm.cost(completion)
        return messages
    
class EvaluationVoting(Node):
    input_state = 'scored'
    output_state = 'scored'
    max_indegree = None

    @retry(max_attempt = 3)
    async def __call__(self, messages_list: List[Messages]) -> Messages:
        if len(messages_list) == 1:
            return messages_list[0]

        messages = messages_list[0].copy()
        random.shuffle(messages_list)
        scores_dict: Dict[str, EvalScores] = {
            chr(65 + idx): msgs.scores
            for idx, msgs in enumerate(messages_list)
        }

        prompt = evaluation_voting_template.format(
            scenario = messages.meta_data['scenario'],
            message = messages.to_json(),
            criteria = messages.meta_data['criteria']
        ) + '\n' + ''.join([f'{choice}. {scores.to_json()}\n' for choice, scores in scores_dict.items()])
        
        completion = await self.llm.get_response(
            messages = [{'role': 'user', 'content': prompt}, ]
        )
        response = completion.choices[0].message.content.strip()

        choice = extract_boxed(response)
        scores = scores_dict[choice]
        scores.source = self.llm.model_name
        messages.scores = scores
        messages.cost[self.name] = self.llm.cost(completion)
        return messages
    
class Debate(Node):
    input_state = 'scored'
    output_state = 'scored'
    max_indegree = None
    
    def __init__(self) -> None:
        super().__init__(None)

    @retry(max_attempt = 3)
    async def __call__(self, messages_list: List[Messages]) -> Messages:
        if len(messages_list) == 1:
            return messages_list[0]
        
        messages = messages_list[0].copy()
        contexts = evaluation_template.format(
            scenario = messages.meta_data['scenario'],
            message = messages.to_json(),
            criteria = messages.meta_data['criteria']
        )

        cost = 0
        for idx in range(len(messages_list)):
            self_response = messages_list[idx].scores.to_json()
            other_responses = messages_list[:idx] + messages_list[idx + 1:]
            other_responses = '\n'.join(
                [json.dumps(msgs.scores.to_json(), ensure_ascii = False) for msgs in other_responses]
            )

            prompt = debate_template.format(
                contexts = contexts,
                self_response = self_response,
                other_responses = other_responses
            )
            llm = get_model(messages_list[idx].scores.source)
            completion = await llm.get_response(
                messages = [{'role': 'user', 'content': prompt}, ]
            )
            cost += llm.cost(completion)
            response = completion.choices[0].message.content.strip()

            scores = extract_json(response)
            Evaluate.check_scores(scores, messages.meta_data['criteria'])
            messages_list[idx].scores = EvalScores([EvalScore(**score) for score in scores])
            # print(response_list[idx].to_json())
        
        for msgs in messages_list:
            msgs.cost[self.name] = cost
        return messages_list