import torch
import os
import json
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
try:
    from vllm import LLM
    from vllm import SamplingParams
except:
    pass
class Base_LLM():

    def __init__(self, model_name: str) -> None:
        
        self.model_name = model_name
        self.client = None

    def get_response(self, **kwargs) -> ChatCompletion:

        raise NotImplementedError
    
    def cost(self, **kwargs) -> float:

        raise NotImplementedError

class LLM_API(Base_LLM):

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        price: dict
    ) -> None:
        super().__init__(model_name)

        self.client = OpenAI(
            api_key = api_key,
            base_url = base_url
        )
        self.price = price

    def get_response(
        self,
        messages: list,
        **kwargs
    ) -> ChatCompletion:

        completion = self.client.chat.completions.create(
            model = self.model_name,
            messages = messages,
            **kwargs
        )

        return completion
    
    def cost(self, completion: ChatCompletion) -> float:

        return completion.usage.prompt_tokens * self.price['prompt'] + \
            completion.usage.completion_tokens * self.price['completion']
    
class LLM_VLLM(Base_LLM):

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        args: dict,
        price: dict = None
    ) -> None:
        super().__init__(model_name)

        # 初始化vllm引擎和生成参数
        self.llm = Base_LLM(
            model = args.model_path,
            tensor_parallel_size = getattr(args, 'tensor_parallel_size', 1),
            dtype = getattr(args, 'dtype', 'auto'),
            trust_remote_code = getattr(args, 'trust_remote_code', False),
            gpu_memory_utilization = getattr(args, 'gpu_memory_utilization', 0.9)
        )
        # 配置生成参数
        self.sampling_params = SamplingParams(
            temperature = getattr(args, 'temperature', 0.5),
            top_p = getattr(args, 'top_p', 0.95),
            max_tokens = getattr(args, 'max_tokens', 256),
            n = getattr(args, 'n', 1),
            stop = getattr(args, 'stop', None),
            presence_penalty = getattr(args, 'presence_penalty', 0.0)
        )
        self.price = price

    @torch.no_grad()
    def get_response(
        self,
        messages: list,
        **kwargs
    ) -> ChatCompletion:

        outputs = self.llm.chat(
            messages,
            self.sampling_params,
            use_tqdm = False
        )

        choice = Choice(
            finish_reason = 'stop',
            index = 0,
            message = ChatCompletionMessage(
                role = 'assistant',
                content = outputs[0].outputs[0].text
            )
        )
        completion = ChatCompletion(
            id = '0',
            choices = [choice],
            created = 1,
            model = self.model_name
        )

        return completion
    
    def cost(self, **kwargs) -> float:

        return 0