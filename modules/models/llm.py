import torch
import os
import json
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from transformers import AutoModelForSequenceClassification, AutoTokenizer
try:
    from vllm import LLM
    from vllm import SamplingParams
except:
    pass

class Base_LLM():

    def __init__(self, model_name: str) -> None:
        
        self.model_name = model_name
        self.client = None

    async def get_response(self, **kwargs) -> ChatCompletion:
        raise NotImplementedError

    async def tool_use(self, **kwargs) -> ChatCompletion:
        raise NotImplementedError

    def get_reward(self, **kwargs) -> int:
        raise NotImplementedError
    
    def cost(self, completion: ChatCompletion, **kwargs) -> float:
        raise NotImplementedError

class LLM_API(Base_LLM):

    def __init__(
        self,
        model_name: str,
        model_name_client: str,
        api_key: str,
        base_url: str,
        price: dict
    ) -> None:
        super().__init__(model_name)

        self.model_name_client = model_name_client
        self.api_key = api_key
        self.base_url = base_url
        self.price = price
        self.client = AsyncOpenAI(
            api_key = api_key,
            base_url = base_url
        )

    async def get_response(
        self,
        messages: list,
        **kwargs
    ) -> ChatCompletion:

        completion = await self.client.chat.completions.create(
            model = self.model_name_client,
            messages = messages,
            **kwargs
        )

        return completion
    
    def cost(self, completion: ChatCompletion, **kwargs) -> float:

        return completion.usage.prompt_tokens * self.price['prompt'] + \
            completion.usage.completion_tokens * self.price['completion']
    
class LLM_VLLM(Base_LLM):

    def __init__(
        self,
        model_name: str,
        model_path: str,
        **kwargs
    ) -> None:
        super().__init__(model_name)

        print(f'cuda devices: {torch.cuda.device_count()}')
        
        self.llm = LLM(
            model = os.path.abspath(model_path),
            tensor_parallel_size = torch.cuda.device_count(),
            dtype = getattr(kwargs, 'dtype', 'auto'),
            trust_remote_code = getattr(kwargs, 'trust_remote_code', False),
            gpu_memory_utilization = getattr(kwargs, 'gpu_memory_utilization', 0.9)
        )
        self.sampling_params = SamplingParams(
            temperature = getattr(kwargs, 'temperature', 0.5),
            top_p = getattr(kwargs, 'top_p', 0.95),
            max_tokens = getattr(kwargs, 'max_tokens', 1024),
            n = getattr(kwargs, 'n', 1),
            stop = getattr(kwargs, 'stop', None),
            presence_penalty = getattr(kwargs, 'presence_penalty', 0.0)
        )

    @torch.no_grad()
    async def get_response(
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
            model = self.model_name,
            object = 'chat.completion'
        )

        return completion
    
    def cost(self, completion: ChatCompletion, **kwargs) -> float:

        return 0
    
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
    
    def cost(self, completion: ChatCompletion, **kwargs) -> float:

        return 0