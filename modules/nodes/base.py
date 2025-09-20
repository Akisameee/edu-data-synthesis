import random
import asyncio
from tqdm import tqdm
from copy import deepcopy
from typing import Callable
from string import Formatter
import aiofiles
from jinja2 import Environment, FileSystemLoader, Template as JinjaTemplate, meta
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool

import sys
sys.path.insert(0, '..')

from modules.models import *
from modules.tools import *
from modules.base import *

CACHE_DIR = './modules/nodes/.cache'

class Template:
    template: JinjaTemplate
    keys: Set[str]
    env: Environment
    
    def __init__(self, template_path: str) -> None:
        template_dir = os.path.dirname(template_path)
        template_file = os.path.basename(template_path)
        self.env = Environment(
            loader = FileSystemLoader(template_dir),
            autoescape = False,
            trim_blocks = True,
            lstrip_blocks = True
        )
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        parsed_content = self.env.parse(template_content)
        self.keys = meta.find_undeclared_variables(parsed_content)
        self.template = self.env.get_template(template_file)
    
    def format(self, messages: Messages, **kwargs) -> str:
        for key in self.keys:
            if key in kwargs:
                continue
            elif key == 'messages':
                kwargs[key] = messages.to_list()
            elif key == 'scenario':
                kwargs[key] = messages.metadata.scenario.to_md(1)
            elif key == 'criteria':
                kwargs[key] = messages.metadata.criteria.to_md(1)
            else:
                raise KeyError(f'Failed to format template with key={key}.')
        missing_vars = self.keys - set(kwargs.keys())
        if missing_vars: raise KeyError(f'Missing variables: {missing_vars}')
        return self.template.render(**kwargs)

# class Template:
#     template: str
#     keys: Set[str]

#     def __init__(self, template_path: str) -> None:
#         with open(template_path, 'r', encoding = 'utf-8') as f:
#             self.template = f.read()
#         formatter = Formatter()
#         self.keys = set()
#         for _, key, _, _ in formatter.parse(self.template):
#             if key is not None:
#                 self.keys.add(key)

#     def format(self, messages: Messages, **kwargs) -> str:
#         for key in self.keys:
#             if key in kwargs:
#                 continue
#             elif key == 'messages':
#                 kwargs[key] = messages.to_list()
#             elif key == 'scenario':
#                 kwargs[key] = messages.metadata.scenario.to_md(1)
#             elif key == 'criteria':
#                 kwargs[key] = messages.metadata.criteria.to_md(1)
#             else:
#                 raise KeyError(f'Failed to format template with key={key}.')
#         return self.template.format(**kwargs)
    
class Node():
    name: Optional[str] = None
    input_state: Optional[MessagesState] = None
    output_state: Optional[MessagesState] = None
    max_indegree: Optional[int] = None
    max_outdegree: Optional[int] = None

    llm: Optional[Base_LLM]
    tools: Optional[List[BaseTool]]

    cache: bool = False
    max_cache_count: int = 5

    def __init__(
        self,
        llm: Optional[str | Base_LLM],
        tools: Optional[List[str]] = []
    ) -> None:
        if llm is not None and isinstance(llm, str):
            llm = get_model(llm)
        self.llm = llm
        if tools is not None:
            tools = get_tools(tools)
        self.tools = tools

        if self.cache:
            self.cache_path = os.path.join(CACHE_DIR, '_'.join(self.to_tuple()))
            os.makedirs(os.path.dirname(self.cache_path), exist_ok = True)
            self._cache_lock = asyncio.Lock()
            self.__call__ = self._cache_decorator(self.max_cache_count)(self.__call__)

    async def _cache_load(self, messages: Messages, max_cache_count: int) -> Optional[Messages]:
        if messages.metadata.id is None:
            return None
        async with self._cache_lock:
            if not os.path.exists(self.cache_path):
                return None
            async with aiofiles.open(self.cache_path, 'r', encoding = 'utf-8') as af:
                cached_dict: Dict[str, List[Dict]] = json.loads(await af.read())
            if messages.metadata.id not in cached_dict:
                return None
            cached_outputs = cached_dict[messages.metadata.id]
            if len(cached_outputs) < max_cache_count:
                return None
            else:
                cached_output = random.choice(cached_outputs)
                for key, value in cached_output.items():
                    if key == 'scores':
                        messages.scores = EvalScores.from_dict(value)
                    elif key == 'cost':
                        messages.cost[self.name] = value
                    else: raise NotImplementedError
                return messages
    
    async def _cache_save(self, messages: Messages) -> None:
        if messages.metadata.id is None:
            return
        async with self._cache_lock:
            if not os.path.exists(self.cache_path):
                async with aiofiles.open(self.cache_path, 'r', encoding = 'utf-8') as af:
                    cached_dict: Dict[str, List[Dict]] = json.loads(await af.read())
            else: cached_dict = {}
            if messages.metadata.id not in cached_dict:
                cached_dict[messages.metadata.id] = []
            cached_output = {}
            if self.llm is not None: cached_output['cost'] = messages.cost[self.name]
            if self.output_state == 'scored':
                cached_output['scores'] = messages.scores.to_dict()
            else: raise NotImplementedError
            cached_dict[messages.metadata.id].append(cached_output)

            async with aiofiles.open(self.cache_path, 'w', encoding = 'utf-8') as af:
                await af.write(json.dumps(cached_dict, ensure_ascii = False, indent = 4))
    
    def _cache_decorator(self, max_cache_count: int = 5) -> Callable:
        def decorator(call_func):
            if call_func.__name__ != '__call__':
                raise ValueError('The cache decorator can only be used on __call__ method.')
            async def wrapper(self: 'Node', messages: Messages, *args, **kwargs):
                cached_messages = await self._cache_load(messages, max_cache_count)
                if cached_messages is not None:
                    return cached_messages
                result = await call_func(self, messages, *args, **kwargs)
                await self._cache_save(result)
                return result
            return wrapper
        return decorator

    async def get_response(self, messages: List[Dict[str, str]], **kwargs) -> Tuple[BaseMessage, float]:
        if self.llm is None:
            raise NotImplementedError
        return await self.llm.get_response(messages, self.tools, **kwargs)

    def to_tuple(self) -> tuple:
        return (
            self.__class__.__name__,
            self.llm.model_name if self.llm is not None else '',
            [tool.name for tool in self.tools]
        )

    def to_dict(self) -> dict:
        return {
            'class_module': self.__class__.__module__,
            'class_name': self.__class__.__name__,
            'model_name': self.llm.model_name if self.llm is not None else None,
            'tools': [tool.name for tool in self.tools],
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
    
    async def __call__(self, inputs: Messages | List[Messages]) -> Any:
        return inputs
        
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
