import torch
import os
from tqdm import tqdm
from openai import OpenAI, AsyncOpenAI
from openai.types.chat.chat_completion import Choice
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from typing import List, Dict, Optional, Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer
# try:
#     from vllm import LLM
#     from vllm import SamplingParams
# except:
#     pass

class Base_LLM():

    def __init__(self, name: str) -> None:
        
        self.name = name
        self.client = None

    async def get_response(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[BaseTool]] = None,
        **kwargs
    ) -> Tuple[BaseMessage, float]:
        raise NotImplementedError

    def get_reward(self, **kwargs) -> int:
        raise NotImplementedError
    
    def get_cost(self, completion: BaseMessage, **kwargs) -> float:
        raise NotImplementedError

class LLM_API(Base_LLM):

    def __init__(
        self,
        name: str,
        name_client: str,
        api_key: str,
        base_url: str,
        price: dict
    ) -> None:
        super().__init__(name)
        self.name_client = name_client
        self.api_key = api_key
        self.base_url = base_url
        self.price = price
        self.client = ChatOpenAI(
            model = name_client,
            openai_api_key = api_key,
            openai_api_base = base_url
        )

    async def get_response(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[BaseTool]] = None,
        **kwargs
    ) -> Tuple[BaseMessage, float]:
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
        client_with_tools = self.client
        if tools:
            client_with_tools = self.client.bind_tools(tools)
        cost = 0
        max_tool_call_steps = kwargs.get('max_tool_call_steps', 5)
        for _ in range(max_tool_call_steps):
            response = await client_with_tools.ainvoke(langchain_messages, **kwargs)
            langchain_messages.append(response)
            cost += self.get_cost(response)
            tool_messages = await self._execute_tools(response, tools)
            if tool_messages:
                langchain_messages += tool_messages
            else: return response, cost
        raise RuntimeError(f'Exceeded max tool call steps: {max_tool_call_steps}.')
    
    async def _execute_tools(self, response: BaseMessage, tools: List[BaseTool]) -> List[ToolMessage]:
        tool_messages = []
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool = next((t for t in tools if t.name == tool_name), None)
                if tool:
                    try:
                        tool_result = await tool.ainvoke(tool_args)
                        content = str(tool_result)
                        # tqdm.write(f'tool: {tool_name}, tool_args: {tool_args}, response: {content[:50]}')
                    except Exception as e:
                        content = f'Error occured when executing {tool_name}: {str(e)}'
                else: content = f'Unknown tool: {tool_name}'
                tool_messages.append(ToolMessage(
                    content = content,
                    tool_call_id = tool_call.get("id"),
                    name = tool_name
                ))
        return tool_messages
    
    def get_cost(self, response: BaseMessage) -> float:
        token_usage = response.response_metadata['token_usage']
        if 'prompt_cache_hit_tokens' in self.price and 'prompt_cache_miss_tokens' in self.price:
            prompt_cost = token_usage['prompt_cache_hit_tokens'] * self.price['prompt_cache_hit_tokens']
            prompt_cost += token_usage['prompt_cache_miss_tokens'] * self.price['prompt_cache_miss_tokens']
        else:
            prompt_cost = token_usage['prompt_tokens'] * self.price['prompt']
        completion_cost = token_usage['completion_tokens'] * self.price['completion']
        return prompt_cost + completion_cost
    
# class LLM_VLLM(Base_LLM):

#     def __init__(
#         self,
#         model_name: str,
#         model_path: str,
#         **kwargs
#     ) -> None:
#         super().__init__(model_name)
#         print(f'cuda devices: {torch.cuda.device_count()}')
#         self.llm = LLM(
#             model = os.path.abspath(model_path),
#             tensor_parallel_size = torch.cuda.device_count(),
#             dtype = getattr(kwargs, 'dtype', 'auto'),
#             trust_remote_code = getattr(kwargs, 'trust_remote_code', False),
#             gpu_memory_utilization = getattr(kwargs, 'gpu_memory_utilization', 0.9)
#         )
#         self.sampling_params = SamplingParams(
#             temperature = getattr(kwargs, 'temperature', 0.5),
#             top_p = getattr(kwargs, 'top_p', 0.95),
#             max_tokens = getattr(kwargs, 'max_tokens', 1024),
#             n = getattr(kwargs, 'n', 1),
#             stop = getattr(kwargs, 'stop', None),
#             presence_penalty = getattr(kwargs, 'presence_penalty', 0.0)
#         )

#     @torch.no_grad()
#     async def get_response(
#         self,
#         messages: list,
#         **kwargs
#     ) -> ChatCompletion:
#         outputs = self.llm.chat(
#             messages,
#             self.sampling_params,
#             use_tqdm = False
#         )
#         choice = Choice(
#             finish_reason = 'stop',
#             index = 0,
#             message = ChatCompletionMessage(
#                 role = 'assistant',
#                 content = outputs[0].outputs[0].text
#             )
#         )
#         completion = ChatCompletion(
#             id = '0',
#             choices = [choice],
#             created = 1,
#             model = self.model_name,
#             object = 'chat.completion'
#         )
#         return completion
    
#     def cost(self, completion: ChatCompletion, **kwargs) -> float:
#         return 0
    
class RM_HF(Base_LLM):

    def __init__(
        self,
        model_name: str,
        model_path: str,
        **kwargs
    ) -> None:
        super().__init__(model_name)

        self.device = 'cuda'
        self.llm = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            # attn_implementation="flash_attention_2",
            num_labels=1,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    @torch.no_grad()
    def get_reward(
        self,
        messages: list,
        **kwargs
    ) -> int:

        messages_formatted = self.tokenizer.apply_chat_template(messages, tokenize=False)
        if self.tokenizer.bos_token is not None and messages_formatted.startswith(self.tokenizer.bos_token):
            messages_formatted = messages_formatted[len(self.tokenizer.bos_token):]
        inputs = self.tokenizer(messages_formatted, return_tensors="pt").to(self.device)

        score = self.llm(**inputs, **kwargs).logits[0][0].item()

        return score
    
    def get_cost(self, completion: BaseMessage) -> float:

        return 0