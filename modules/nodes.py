from tqdm import tqdm
from copy import deepcopy
import random
import functools
from typing import Literal, List, Dict, Generic, TypeVar, ClassVar
from dataclasses import dataclass, field

import sys

from models import Base_LLM
sys.path.insert(0, '.')

from models import *
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
    def __add__(self, other: 'GenericList[T]'): return GenericList(self._items + other._items)

@dataclass
class Message:
    role: Literal['system', 'user', 'assistant']
    content: str

class Messages(GenericList[Message]):
    source: Base_LLM = None
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
    score: int | float
    reason: str

class EvalScores(GenericList[EvalScore]):
    source: Base_LLM = None
    messages: Messages = None

    def sum(self) -> float:
        return sum([score.score for score in self._items])

    def get_score(self, criterion: str) -> Optional[EvalScore]:
        for score in self._items:
            if score.criterion == criterion:
                return score
        return None

    def update(self, other: 'EvalScores') -> None:
        criterion_idxs = {scores.criterion: idx for idx, scores in enumerate(self._items)}
        for scores in other:
            if scores.criterion in criterion_idxs:
                self._items[criterion_idxs[scores.criterion]] = scores
            else:
                self.append(scores)

class Node():
    parents: List['Node']
    children: List['Node']
    input_type = None
    output_type = None
    max_indegree = None

    def __init__(self, llm: Base_LLM = None) -> None:
        self.parents = []
        self.children = []
        self.llm = llm
        self.history = {}

    def run(self, **kwargs) -> Any:
        
        if len(self.parents) == 0:
            return self.__call__(**kwargs)
        else:
            if self.max_indegree is None or self.max_indegree > 1:
                assert all(isinstance(p, Node) for p in self.parents)
                node_input = [parent.run(**kwargs) for parent in self.parents]
                return self.__call__(node_input, **kwargs)
            elif self.max_indegree == 1:
                node_input = self.parents[0].run(**kwargs)
                return self.__call__(node_input, **kwargs)
            else:
                raise ValueError('Invalid parent')
        
    def __call__(self, **kwargs) -> Any:
        pass
    
class SystemGenerate(Node):
    output_type = SystemMessages

    def __init__(self, llm: Base_LLM = None) -> None:
        super().__init__(llm)

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

    def __init__(self, llm: Base_LLM = None) -> None:
        super().__init__(llm)

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

    def __init__(self, llm: Base_LLM = None) -> None:
        super().__init__(llm)

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

    def __init__(self, llm: Base_LLM = None) -> None:
        super().__init__(llm)

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
    max_indegree = 1

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
        # AI_EVALUATOR_PROMPT_ZH
        prompt = evaluation_template.format(
            scenario = kwargs['scenario'],
            message = messages.to_json(),
            criteria = kwargs['criteria']
        )
        # prompt = AI_EVALUATOR_PROMPT_ZH.format(
        #     scenario = kwargs['scenario'],
        #     message = messages.to_json(),
        #     criteria = kwargs['criteria']
        # )

        completion = self.llm.get_response(
            messages = [{'role': 'user', 'content': prompt}, ],
            temperature = 0.0
        )
        self.llm.cost(completion)
        response = completion.choices[0].message.content.strip()

        scores = extract_json(response)
        self.check_scores(scores, kwargs['criteria'])
        scores = EvalScores([EvalScore(**score) for score in scores])
        scores.source = self.llm
        scores.messages = messages
        return scores
    
class EvaluateICL(Node):
    input_type = AssistantMessages
    output_type = EvalScores
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
    def __call__(
        self,
        messages: AssistantMessages,
        **kwargs
    ) -> EvalScores:

        prompt = evaluation_cl_template.format(
            scenario = kwargs['scenario'],
            message = messages.to_json(),
            criteria = kwargs['criteria'],
            samples = self.get_human_eval_sample(
                kwargs['task'],
                kwargs['id']
            )
        )

        completion = self.llm.get_response(
            messages = [{'role': 'user', 'content': prompt}, ],
            temperature = 0.0
        )
        self.llm.cost(completion)
        response = completion.choices[0].message.content.strip()

        scores = extract_json(response)
        Evaluate.check_scores(scores, kwargs['criteria'])
        scores = EvalScores([EvalScore(**score) for score in scores])
        scores.source = self.llm
        scores.messages = messages
        return scores


class EvaluateSingle(Node):
    input_type = AssistantMessages
    output_type = EvalScores
    max_indegree = 1

    @retry(max_attempt = 3)
    def __call__(
        self,
        messages: AssistantMessages,
        **kwargs
    ) -> EvalScores:

        if isinstance(self.llm, (LLM_API, LLM_VLLM)):
            return self._call_llm_api(messages, **kwargs)
        elif isinstance(self.llm, RM_HF):
            return self._call_rm_hf(messages, **kwargs)
    
    def _call_llm_api(
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
        scores.source = self.llm
        scores.messages = messages
        return scores
    
    def _call_rm_hf(
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

        Evaluate.check_scores(scores, kwargs['criteria'])
        scores = EvalScores([EvalScore(**score) for score in scores])
        scores.source = self.llm
        scores.messages = messages
        return scores
    
class EvaluationAggregation(Node):
    input_type = List[EvalScores]
    output_type = EvalScores
    max_indegree = None

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
        scores.source = self.llm
        scores.messages = scores_list[0].messages
        return scores
    
class EvaluationVoting(Node):
    input_type = List[EvalScores]
    output_type = EvalScores
    max_indegree = None

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
        scores = scores_dict[choice]
        scores.source = self.llm
        return scores
    
class Debate(Node):
    input_type = List[EvalScores] | List[Messages]
    output_type = List[EvalScores] | List[Messages]
    max_indegree = None
    
    def __init__(self, llm: Base_LLM = None) -> None:
        super().__init__(llm)

    @retry(max_attempt = 3)
    def __call__(
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
    input_type = AssistantMessages | EvalScores
    output_type = AssistantMessages | EvalScores
    max_indegree = 1

    def __init__(self) -> None:
        super().__init__(None)

    def __call__(
        self,
        output: AssistantMessages | EvalScores,
        **kwargs
    ) -> AssistantMessages | EvalScores:
        return output
    
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