import sys
sys.path.insert(0, '..')

from modules.nodes.base import *
from modules.nodes.prompt_templates import *
from modules.nodes.evaluate import Evaluate

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
        self.llm.cost(completion)
        response = completion.choices[0].message.content.strip()

        scores = extract_json(response)
        Evaluate.check_scores(scores, messages.meta_data['criteria'])
        scores = EvalScores([EvalScore(**score) for score in scores])
        scores.source = self.llm
        messages.scores = scores
        return messages
    
class EvaluationVoting(Node):
    input_state = 'scored'
    output_state = 'scored'
    max_indegree = None

    @retry(max_attempt = 3)
    async def __call__(
        self,
        scores_list: List[EvalScores],
        **kwargs
    ) -> EvalScores:
        n_scores = len(scores_list)
        if n_scores == 1:
            return scores_list[0]
        
        assert all(scores.messages == scores_list[0].messages for scores in scores_list)
        random.shuffle(scores_list)
        scores_dict = {chr(65 + idx): scores for idx, scores in enumerate(scores_list)}

        prompt = evaluation_voting_template.format(
            scenario = kwargs['scenario'],
            message = scores_list[0].messages.to_json(),
            criteria = kwargs['criteria']
        ) + '\n' + ''.join([f'{choice}. {scores.to_json()}\n' for choice, scores in scores_dict.items()])
        
        completion = await self.llm.get_response(
            messages = [{'role': 'user', 'content': prompt}, ]
        )
        self.llm.cost(completion)
        response = completion.choices[0].message.content.strip()

        choice = extract_boxed(response)
        scores = scores_dict[choice]
        scores.source = self.llm
        return scores
    
class Debate(Node):
    input_state = 'scored'
    output_state = 'scored'
    max_indegree = None
    
    def __init__(self, llm: Base_LLM = None) -> None:
        super().__init__(llm)

    @retry(max_attempt = 3)
    async def __call__(
        self,
        response_list: List[EvalScores] | List[Messages],
        n_round: int = 1,
        **kwargs
    ) -> List[EvalScores] | List[Messages]:
        n_response = len(response_list)
        if n_response == 1:
            return response_list
        
        if isinstance(response_list[0], EvalScores):
            assert all(scores.messages == response_list[0].messages for scores in response_list)
            contexts = evaluation_template.format(
                scenario = kwargs['scenario'],
                message = response_list[0].messages.to_json(),
                criteria = kwargs['criteria']
            )
        elif isinstance(response_list[0], Messages):
            assert all(messages[:-1] == response_list[0][:-1] for messages in response_list)
            contexts = response_list[0][:-1].to_json()

        for round in range(n_round):
            for idx in range(len(response_list)):
                response = response_list[idx]
                # print(response_list[idx].to_json())
                if isinstance(response, EvalScores):
                    self_response = response.to_json()
                    other_responses = response_list[:idx] + response_list[idx + 1:]
                    other_responses = '\n'.join(
                        [json.dumps(r.to_json(), ensure_ascii = False) for r in other_responses]
                    )
                elif isinstance(response, Messages):
                    pass

                prompt = debate_template.format(
                    contexts = contexts,
                    self_response = self_response,
                    other_responses = other_responses
                )
                llm = response.source
                completion = response.source.get_response(
                    messages = [{'role': 'user', 'content': prompt}, ]
                )
                response.source.cost(completion)
                response_json = extract_json(completion.choices[0].message.content.strip())
                if isinstance(response, EvalScores):
                    Evaluate.check_scores(response_json, kwargs['criteria'])
                    response_list[idx] = EvalScores([EvalScore(**score) for score in response_json])
                    response_list[idx].source = response.source
                    response_list[idx].messages = response.messages
                elif isinstance(response, Messages):
                    pass
                # print(response_list[idx].to_json())
            
        return response_list