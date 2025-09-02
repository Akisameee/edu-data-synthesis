from tqdm import tqdm
from copy import deepcopy
import random
import functools
from typing import Literal, List, Dict, Generic, TypeVar, ClassVar
from dataclasses import dataclass, field

import sys
sys.path.insert(0, '..')

from modules.models import *
from modules.base import *
from modules.utils import *

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

    def check_parent(self, parent: 'Node') -> bool:
        pass
    
    def check_child(self, child: 'Node') -> bool:
        pass

    async def run(self, **kwargs) -> Any:
        
        if len(self.parents) == 0:
            return await self.__call__(**kwargs)
        else:
            if self.max_indegree is None or self.max_indegree > 1:
                assert all(isinstance(p, Node) for p in self.parents)
                node_input = [await parent.run(**kwargs) for parent in self.parents]
                return await self.__call__(node_input, **kwargs)
            elif self.max_indegree == 1:
                node_input = await self.parents[0].run(**kwargs)
                return await self.__call__(node_input, **kwargs)
            else:
                raise ValueError('Invalid parent')
        
    async def __call__(self, **kwargs) -> Any:
        pass
    
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
