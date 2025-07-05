from tqdm import tqdm
from copy import deepcopy
import random
import functools
from typing import Literal, List, Dict, Generic, TypeVar, ClassVar
from dataclasses import dataclass, field

import sys
sys.path.insert(0, '.')

from models import Base_LLM, get_model
from modules.state import SynthesisState
from modules.prompt_templates import *
from modules.utils import *

T = TypeVar('T')

class GenericList(Generic[T]):

    def __init__(self, items: List[T]): self._items = items
    def __len__(self) -> int: return len(self._items)
    def __getitem__(self, idx: int) -> T: return self._items[idx]
    def __iter__(self): yield from self._items
    def append(self, item: T) -> None: self._items.append(item)
    def pop(self, idx: int = -1) -> T: return self._items.pop(idx)
    def to_json(self) -> list: return [_item.__dict__ for _item in self._items]
    def __eq__(self, other: 'GenericList[T]'): return all(a == b for a, b in zip(self._items, other._items))

@dataclass
class Message:
    role: Literal['system', 'user', 'assistant']
    content: str

class Messages(GenericList[Message]):
    ROLE_TO_CLASS: ClassVar[dict] = {}

    def __init__(self, messages: List[Message]):
        self._items = messages.copy()
        self._auto_convert = True
        self._check_and_convert()
    
    def _check_and_convert(self) -> None:
        if not self._auto_convert or not self._items:
            return
        last_role = self._items[-1].role
        target_class = self.ROLE_TO_CLASS.get(last_role)
        if target_class and not isinstance(self, target_class):
            self.__class__ = target_class
    
    def append(self, message: Message) -> None:
        self._items.append(message)
        self._check_and_convert()
    
    def pop(self, idx: int = -1) -> Message:
        msg = self._items.pop(idx)
        self._check_and_convert()
        return msg

class SystemMessages(Messages):
    def __init__(self, messages: List[Message]):
        super().__init__(messages)

class UserMessages(Messages):
    def __init__(self, messages: List[Message]):
        super().__init__(messages)

class AssistantMessages(Messages):
    def __init__(self, messages: List[Message]):
        super().__init__(messages)

Messages.ROLE_TO_CLASS = {
    'system': SystemMessages,
    'user': UserMessages,
    'assistant': AssistantMessages
}

@dataclass
class EvalScore:
    criterion: str
    score: int
    reason: str

class EvalScores(GenericList[EvalScore]):
    messages: Messages = None

class Node():
    parents = []
    children = []
    input_type = None
    output_type = None

    def __init__(self, llm: Base_LLM = None) -> None:
        self.llm = llm
    
class SystemGenerate(Node):
    output_type = SystemMessages

    def __call__(
        self,
        **kwargs
    ) -> SystemMessages:
        
        system_prompt = system_template.format(
            task = kwargs['scenario']['task'],
            criteria = '\n'.join([c['metric'] for c in kwargs['criteria']])
        )
        return Messages([Message(role = 'system', content = system_prompt)])

class UserGenerate(Node):
    input_type = (SystemMessages, AssistantMessages)
    output_type = UserMessages

    @staticmethod
    def replace_meta_data(content: str, meta_data: str):

        if '[meta_data]' in content:
            content = content.replace('[meta_data]', meta_data, 1)
        else:
            content = f'{meta_data}\n{content}'
        return content

    @retry(max_attempt = 3)
    def __call__(
        self,
        messages: Messages,
        **kwargs
    ) -> UserMessages:
        
        prompt = user_generate_template.format(
            scenario = kwargs['scenario'],
            meta_data = kwargs['meta_data'],
            message = messages.to_json()
        )

        completion = self.llm.get_response(
            messages = [{'role': 'user', 'content': prompt}, ]
        )
        self.llm.cost(completion)
        response = completion.choices[0].message.content.strip()
        
        json_obj = extract_json(response)
        assert json_obj['role'] == 'user'
        assert 'role' in json_obj and 'content' in json_obj
        assert json_obj['role'] == 'user'
        
        messages.append(Message(
            role = 'user',
            content = self.replace_meta_data(
                json_obj['content'], kwargs['meta_data']
            )
        ))
        return messages
    
class AssistantGenerate(Node):
    input_type = UserMessages
    output_type = AssistantMessages

    @retry(max_attempt = 3)
    def __call__(
        self,
        messages: UserMessages,
        **kwargs
    ) -> AssistantMessages:
        
        completion = self.llm.get_response(messages = messages.to_json())
        self.llm.cost(completion)
        response = completion.choices[0].message.content.strip()

        messages.append(Message(
            role = 'assistant',
            content = response
        ))
        return messages
    
class ResponseAggregate(Node):
    input_type = List[AssistantMessages]
    output_type = AssistantMessages

    @retry(max_attempt = 3)
    def __call__(
        self,
        messages_list: List[AssistantMessages],
        **kwargs
    ) -> AssistantMessages:
        n_messages = len(messages_list)
        if n_messages == 1:
            return messages_list[0]
        
        for i in range(n_messages):
            if any(msg != messages_list[i][0] for msg in messages_list[i]):
                break

        history = messages_list[0].to_json()[:i]
        responses = [messages[-1] for messages in messages_list]

        prompt = response_aggregate_template.format(
            scenario = kwargs['scenario'],
            history = history
        ) + '\n' + ''.join([f'Response {idx}:\n{response}\n' for idx, response in enumerate(responses)])
        
        completion = self.llm.get_response(
            messages = [{'role': 'user', 'content': prompt}, ]
        )
        self.llm.cost(completion)
        response = completion.choices[0].message.content.strip()

        json_obj = extract_json(response)
        assert 'role' in json_obj and 'content' in json_obj
        assert json_obj['role'] == 'assistant'

        history.append(Message(
            role = 'assistant',
            content = json_obj['content']
        ))
        return history
    
class Evaluate(Node):
    input_type = AssistantMessages
    output_type = EvalScores

    @staticmethod
    def check_scores(scores: list, criteria: list):

        extra_criteria = []
        for score in scores:
            criterion = [c['metric'] for c in criteria if score['criterion'] in c['metric']]
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
            set(c['metric'] for c in criteria):
            invalid_criteria = [score['criterion'] for score in scores]
            required_criteria = [c['metric'] for c in criteria]
            raise ValueError(f'[Score Parse Error] Invalid criteria: {invalid_criteria}, required: {required_criteria}.')
        
        return scores
    
    @retry(max_attempt = 3)
    def __call__(
        self,
        messages: AssistantMessages,
        **kwargs
    ) -> EvalScores:

        prompt = evaluation_template.format(
            scenario = kwargs['scenario'],
            message = messages.to_json(),
            criteria = kwargs['criteria']
        )

        completion = self.llm.get_response(
            messages = [{'role': 'user', 'content': prompt}, ],
            temperature = 0.0
        )
        self.llm.cost(completion)
        response = completion.choices[0].message.content.strip()

        scores = extract_json(response)
        self.check_scores(scores, kwargs['criteria'])
        scores = EvalScores([EvalScore(**score) for score in scores])
        scores.messages = messages
        return scores

class EvaluateSingle(Node):
    input_type = AssistantMessages
    output_type = EvalScores

    @retry(max_attempt = 3)
    def __call__(
        self,
        messages: AssistantMessages,
        **kwargs
    ) -> EvalScores:

        scores = []
        for criterion in kwargs['criteria']:
            prompt = evaluation_single_template.format(
                scenario = kwargs['scenario'],
                message = messages.to_json(),
                criterion = criterion
            )

            completion = self.llm.get_response(
                messages = [{'role': 'user', 'content': prompt}, ],
                temperature = 0.0
            )
            self.llm.cost(completion)
            response = completion.choices[0].message.content.strip()
            scores.append(extract_json(response))

        Evaluate.check_scores(scores, kwargs['criteria'])
        scores = EvalScores([EvalScore(**score) for score in scores])
        scores.messages = messages
        return scores
    
class EvaluationAggregation(Node):
    input_type = List[EvalScores]
    output_type = EvalScores

    @retry(max_attempt = 3)
    def __call__(
        self,
        scores_list: List[EvalScores],
        **kwargs
    ) -> EvalScores:
        n_scores = len(scores_list)
        if n_scores == 1:
            return scores_list[0]
        
        assert all(scores.messages == scores_list[0].messages for scores in scores_list)

        prompt = evaluation_aggregate_template.format(
            scenario = kwargs['scenario'],
            message = scores_list[0].messages.to_json(),
            criteria = kwargs['criteria']
        ) + '\n' + ''.join([f'Scores {idx}:\n{scores.to_json()}\n' for idx, scores in enumerate(scores_list)])
        
        completion = self.llm.get_response(
            messages = [{'role': 'user', 'content': prompt}, ]
        )
        self.llm.cost(completion)
        response = completion.choices[0].message.content.strip()

        scores = extract_json(response)
        Evaluate.check_scores(scores, kwargs['criteria'])
        scores = EvalScores([EvalScore(**score) for score in scores])
        scores.messages = scores_list[0].messages
        return scores
    
class EvaluationVoting(Node):
    input_type = List[EvalScores]
    output_type = EvalScores

    @retry(max_attempt = 3)
    def __call__(
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
        
        completion = self.llm.get_response(
            messages = [{'role': 'user', 'content': prompt}, ]
        )
        self.llm.cost(completion)
        response = completion.choices[0].message.content.strip()

        choice = extract_boxed(response)
        return scores_dict[choice]
    
# class Review(Node):
#     input_type = AssistantMessages
#     output_type = EvalScores

#     @retry(max_attempt = 3)
#     def __call__(
#         self,
#         state: SynthesisState,
#         llm: Base_LLM
#     ) -> SynthesisState:
#         self.check_required_keys(state)

#         message = deepcopy(state.message)
#         if message[0]['role'] == 'system':
#             message = message[1:]

#         prompt = review_template.format(
#             scenario = state.scenario,
#             message = message,
#             criteria = state.criteria
#         )

#         messages = [{'role': 'user', 'content': prompt}, ]
#         completion = llm.get_response(messages = messages)
#         state.cost += llm.cost(completion)
#         response = completion.choices[0].message.content.strip()

#         critique = extract_json(response)
#         state.critique = critique
        
#         return state
    
class Refine(Node):

    required_keys = ('scenario', 'message_assistant', 'critique')
    description = 'Refine message with critique'

    @retry(max_attempt = 3)
    def __call__(
        self,
        state: SynthesisState,
        llm: Base_LLM
    ) -> SynthesisState:
        self.check_required_keys(state)

        message_dict = {
            str(idx): m for idx, m in enumerate(state.message)
            if m['role'] != 'system'
        }
        assistant_idxs = [
            idx for idx, m in message_dict.items()
            if m['role'] == 'assistant'
        ]

        prompt = refine_template.format(
            scenario = state.scenario,
            message = message_dict,
            assistant_idxs = assistant_idxs,
            critique = state.critique
        )
        
        messages = [{'role': 'user', 'content': prompt}, ]
        completion = llm.get_response(messages = messages)
        state.cost += llm.cost(completion)
        response = completion.choices[0].message.content.strip()
        
        refined_dict: dict = extract_json(response)
        if all(idx not in assistant_idxs for idx in refined_dict.keys()):
            raise ValueError(f'[Refine Error] Invalid message indexs: {refined_dict.keys()}.')

        for idx, refined_m in refined_dict.items():
            assert refined_m['role'] == 'assistant'
            if state.message[int(idx)]['content'] == refined_m['content']:
                raise ValueError('[Refine Error] No content changes.')
            state.message[int(idx)]['content'] = refined_m['content']
        
        state.scores = None
        state.critique = None

        return state
    
class Output(Node):
    input_type = AssistantMessages
    output_type = AssistantMessages

    def __call__(
        self,
        message: AssistantMessages
    ) -> AssistantMessages:
        return message
    
if __name__ == '__main__':

    test_args = {
        'scenario': {
            "task": "回答问题",
            "description": "user给出一个题目，assistant解答题目"
        },
        'criteria': read_criterias('./data/criteria')['回答问题'],
        'meta_data': "学科:生物\n学制级别:小学\n问题：请简述为什么植物需要阳光才能生长？\n"
    }
    models = ['deepseek-v3', 'deepseek-r1', 'gpt-4o', 'qwen-max']
    models = [get_model(model) for model in models]

    node = SystemGenerate()
    messages = node(**test_args)
    print(messages.__class__)

    node = UserGenerate(models[0])
    messages = node(messages, **test_args)
    print(messages.__class__)
    print(messages.to_json())

    node = AssistantGenerate(models[0])
    messages = node(messages, **test_args)
    print(messages.__class__)
    print(messages.to_json())

    scores_list = []
    for model in models:
        node = Evaluate(model)
        scores = node(messages, **test_args)
        print(scores.__class__)
        print(scores.to_json())
        scores_list.append(scores)

    node = EvaluationAggregation(models[0])
    scores = node(scores_list, **test_args)
    print(scores.__class__)
    print(scores.to_json())