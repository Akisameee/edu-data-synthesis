from tqdm import tqdm
from copy import deepcopy
from string import Formatter
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool

import sys
sys.path.insert(0, '..')

from modules.models import *
from modules.tools import *
from modules.base import *

class Template:
    template: str
    keys: Set[str]

    def __init__(self, template_path: str) -> None:
        with open(template_path, 'r', encoding = 'utf-8') as f:
            self.template = f.read()
        formatter = Formatter()
        self.keys = set()
        for _, key, _, _ in formatter.parse(self.template):
            if key is not None:
                self.keys.add(key)

    def format(self, messages: Messages, **kwargs) -> str:
        for key in self.keys:
            if key in kwargs:
                continue
            elif key == 'messages':
                kwargs[key] = messages.to_json()
            elif key == 'scenario':
                kwargs[key] = messages.metadata.scenario.to_md(1)
            elif key == 'criteria':
                kwargs[key] = messages.metadata.criteria.to_md(1)
            else:
                raise KeyError(f'Failed to format template with key={key}.')
        return self.template.format(**kwargs)
    
class Node():
    name: Optional[str] = None
    input_state: Optional[MessagesState] = None
    output_state: Optional[MessagesState] = None
    max_indegree: Optional[int] = None
    max_outdegree: Optional[int] = None

    llm: Optional[Base_LLM]
    tools: Optional[List[BaseTool]]

    def __init__(
        self,
        llm: Optional[str | Base_LLM],
        tools: Optional[List[str]] = None
    ) -> None:
        if llm is not None and isinstance(llm, str):
            llm = get_model(llm)
        self.llm = llm
        if tools is not None:
            tools = get_tools(tools)
        self.tools = tools

    async def get_response(self, messages: List[Dict[str, str]], **kwargs) -> BaseMessage:
        if self.llm is None:
            raise NotImplementedError
        return await self.llm.get_response(messages, self.tools)

    def to_tuple(self) -> tuple:
        return (
            self.__class__.__name__,
            self.llm.model_name if self.llm is not None else '',
            [tool.name for tool in self.tools] if self.tools is not None else ''
        )

    def to_dict(self) -> dict:
        return {
            'class_module': self.__class__.__module__,
            'class_name': self.__class__.__name__,
            'model_name': self.llm.model_name if self.llm is not None else None,
            'tools': [tool.name for tool in self.tools] if self.tools is not None else None,
            'name': self.name
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Node':
        module_name = data['class_module']
        class_name = data['class_name']
        module = __import__(module_name, fromlist=[class_name])
        node_class = getattr(module, class_name)
        node: Node = node_class(llm = data['model_name'], tools = data['tools'])
        node.name = data['name']
        return node
        
    async def __call__(self, messages: Messages) -> Any:
        return messages
    
# class Review(Node):
#     input_type = AssistantMessages
#     output_type = EvalScores

#     @retry(max_attempt = 3)
#     async def __call__(
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
    
# class Refine(Node):

#     required_keys = ('scenario', 'message_assistant', 'critique')
#     description = 'Refine message with critique'

#     @retry(max_attempt = 3)
#     async def __call__(
#         self,
#         state: SynthesisState,
#         llm: Base_LLM
#     ) -> SynthesisState:
#         self.check_required_keys(state)

#         message_dict = {
#             str(idx): m for idx, m in enumerate(state.message)
#             if m['role'] != 'system'
#         }
#         assistant_idxs = [
#             idx for idx, m in message_dict.items()
#             if m['role'] == 'assistant'
#         ]

#         prompt = refine_template.format(
#             scenario = state.scenario,
#             message = message_dict,
#             assistant_idxs = assistant_idxs,
#             critique = state.critique
#         )
        
#         messages = [{'role': 'user', 'content': prompt}, ]
#         completion = llm.get_response(messages = messages)
#         state.cost += llm.cost(completion)
#         response = completion.choices[0].message.content.strip()
        
#         refined_dict: dict = extract_json(response)
#         if all(idx not in assistant_idxs for idx in refined_dict.keys()):
#             raise ValueError(f'[Refine Error] Invalid message indexs: {refined_dict.keys()}.')

#         for idx, refined_m in refined_dict.items():
#             assert refined_m['role'] == 'assistant'
#             if state.message[int(idx)]['content'] == refined_m['content']:
#                 raise ValueError('[Refine Error] No content changes.')
#             state.message[int(idx)]['content'] = refined_m['content']
        
#         state.scores = None
#         state.critique = None

#         return state
