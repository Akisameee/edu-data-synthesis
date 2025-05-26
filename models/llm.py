import torch
import os
import json
from openai import OpenAI
from openai.types.chat import ChatCompletion

class LLM():

    def __init__(self, model_name: str) -> None:
        
        self.model_name = model_name
        self.client = None

    def get_response(self, **kwargs) -> ChatCompletion:

        raise NotImplementedError

class LLM_API(LLM):

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str
    ) -> None:
        super().__init__(model_name)

        self.client = OpenAI(
            api_key = api_key,
            base_url = base_url
        )

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