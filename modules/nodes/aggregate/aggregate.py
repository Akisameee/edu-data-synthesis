import numpy as np
import random

import sys
sys.path.insert(0, '..')

from modules.nodes.base import *
from modules.nodes.utils import *
from modules.nodes.prompt_templates import *

aggregate_system_template = Template('./modules/nodes/aggregate/aggregate_system.md')
aggregate_user_template = Template('./modules/nodes/aggregate/aggregate_user.md')

class EvaluationAverage(Node):
    input_state = 'scored'
    output_state = 'scored'
    max_indegree = None

    def __init__(self) -> None:
        super().__init__(None)

    @retry(max_attempt = 3)
    async def run(self, messages_list: List[Messages]) -> Messages:
        if len(messages_list) == 1:
            return messages_list[0]
        
        messages = messages_list[0]
        scores_avg = []
        for criterion in messages.scores.names:
            scores_avg.append(EvalScore(
                criterion = criterion,
                score = sum([
                    msgs.scores[criterion].score for msgs in messages_list
                ]) / len(messages_list),
                reason = '\n'.join([
                    msgs.scores[criterion].reason for msgs in messages_list
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
    async def run(self, messages_list: List[Messages]) -> Messages:
        if len(messages_list) == 1:
            return messages_list[0]
        
        messages = messages_list[0]
        scores_max = []
        for criterion in messages.scores.names:
            max_idx = np.argmax([msgs.scores[criterion].score for msgs in messages_list])
            scores_max.append(EvalScore(
                criterion = criterion,
                score = messages_list[max_idx].scores[criterion].score,
                reason = messages_list[max_idx].scores[criterion].reason
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
    async def run(self, messages_list: List[Messages]) -> Messages:
        if len(messages_list) == 1:
            return messages_list[0]
        
        messages = messages_list[0]
        scores_min = []
        for criterion in messages.scores.names:
            max_idx = np.argmin([msgs.scores[criterion].score for msgs in messages_list])
            scores_min.append(EvalScore(
                criterion = criterion,
                score = messages_list[max_idx].scores[criterion].score,
                reason = messages_list[max_idx].scores[criterion].reason
            ))
        messages.scores = EvalScores(scores_min)
        return messages

class EvaluationAggregation(Node):
    input_state = 'scored'
    output_state = 'scored'
    max_indegree = None

    @retry(max_attempt = 3)
    async def run(self, messages_list: List[Messages]) -> Messages:
        if len(messages_list) == 1:
            return messages_list[0]
        messages = messages_list[0].deepcopy()
        system_prompt = aggregate_system_template.format(messages)
        user_prompt = aggregate_user_template.format(
            messages, evaluations = '\n'.join([
                f'  - [{idx}]\n{msgs.to_md(1)}'
                for idx, msgs in enumerate(messages_list)
            ])
        )
        # prompt = evaluation_aggregate_template.format(
        #     scenario = messages.metadata.scenario.__dict__,
        #     message = messages.to_json(),
        #     criteria = messages.metadata.criteria.to_json()
        # ) + '\n' + ''.join([
        #     f'Scores {idx}:\n{msgs.scores.to_json()}\n'
        #     for idx, msgs in enumerate(messages_list)
        # ])
        with open('test_prompt.md', 'w') as f:
            f.write(user_prompt)
        
        response, cost = await self.get_response(
           messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ]
        )
        messages.cost[self.name] = cost
        scores = extract_json(response.content.strip())
        scores = check_scores(scores, messages.metadata.criteria)

        scores.source = self.llm.name
        messages.scores = scores
        return messages
    
class EvaluationVoting(Node):
    input_state = 'scored'
    output_state = 'scored'
    max_indegree = None

    @retry(max_attempt = 3)
    async def run(self, messages_list: List[Messages]) -> Messages:
        if len(messages_list) == 1:
            return messages_list[0]

        messages = messages_list[0].deepcopy()
        random.shuffle(messages_list)
        scores_dict: Dict[str, EvalScores] = {
            chr(65 + idx): msgs.scores
            for idx, msgs in enumerate(messages_list)
        }

        prompt = evaluation_voting_template.format(
            scenario = messages.metadata.scenario.__dict__,
            message = messages.to_list(),
            criteria = messages.metadata.criteria.to_list()
        ) + '\n' + ''.join([f'{choice}. {scores.to_list()}\n' for choice, scores in scores_dict.items()])
        
        response, cost = await self.get_response(
            messages = [{'role': 'user', 'content': prompt}, ]
        )
        messages.cost[self.name] = cost
        choice = extract_boxed(response.content.strip())
        scores = scores_dict[choice]

        scores.source = self.llm.name
        messages.scores = scores
        return messages
    
class Debate(Node):
    input_state = 'scored'
    output_state = 'scored'
    max_indegree = None
    
    def __init__(self) -> None:
        super().__init__(None)

    @retry(max_attempt = 3)
    async def run(self, messages_list: List[Messages]) -> Messages:
        if len(messages_list) == 1:
            return messages_list[0]
        
        messages = messages_list[0].deepcopy()
        contexts = evaluation_template.format(
            scenario = messages.metadata.scenario.__dict__,
            message = messages.to_list(),
            criteria = messages.metadata.criteria.to_list()
        )

        cost = 0
        for idx in range(len(messages_list)):
            self_response = messages_list[idx].scores.to_list()
            other_responses = messages_list[:idx] + messages_list[idx + 1:]
            other_responses = '\n'.join(
                [json.dumps(msgs.scores.to_list(), ensure_ascii = False) for msgs in other_responses]
            )

            prompt = debate_template.format(
                contexts = contexts,
                self_response = self_response,
                other_responses = other_responses
            )
            llm = get_model(messages_list[idx].scores.source)
            response = await llm.get_response(
                messages = [{'role': 'user', 'content': prompt}, ]
            )
            cost += llm.get_cost(response)
            scores = extract_json(response.content.strip())
            messages_list[idx].scores = (scores, messages.metadata.criteria.to_list())
            # print(response_list[idx].to_json())
        
        for msgs in messages_list:
            msgs.cost[self.name] = cost
        return messages_list