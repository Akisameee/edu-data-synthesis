from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
import os
import json
from openai import OpenAI

class LLM():

    def __init__(self, model_name: str) -> None:
        
        self.model_name = model_name
        self.client = OpenAI(
            api_key = "sk-s17gKfBLOAaXHFKGGGF4u4iPVHXicKWtLJqvROqltd17XwLR",
            base_url = "https://api.chatanywhere.tech/v1"
            # base_url="https://api.chatanywhere.org/v1"
        )

    def get_response(
        self,
        prompt
    ):

        completion = self.client.chat.completions.create(
            model = self.model_name,
            messages = [{'role': 'user','content': prompt},]
        )
        return completion.choices[0].message.content.strip()
    
    def function_calling(  
        self,  
        prompt: str,  
        functions: list = None,  
        available_functions: dict = None,  
        function_call = "auto",  
        memory_messages: list = None,  
        max_function_calls: int = 3  
    ):  
        """  
        高级版获取回复，支持函数链式推理 + 多轮对话记忆。  
    
        Args:  
            prompt (str): 用户输入  
            functions (list, optional): 可以调用的 functions 定义  
            available_functions (dict, optional): 本地可用函数 {'func_name': callable}  
            function_call (str/dict, optional): function_call 策略 'auto' / 'none' / {'name': x}  
            memory_messages (list, optional): 保持上下文的历史 messages  
            max_function_calls (int): 允许最大 function_call 次数，防止死循环  

        Returns:  
            dict:  
                - "final_message": 最后生成的 message  
                - "all_messages": 全过程 messages 记录  
                - "function_call_steps": 中间每次 function 调用日志  
        """  
        
        if memory_messages is None:  
            memory_messages = []  

        # 初始 messages  
        messages = memory_messages + [{'role': 'user', 'content': prompt}]  
        function_call_steps = []  # 用于记录 function 调用轨迹  

        for call_iteration in range(max_function_calls + 1):  # 防御死循环  
            payload = {  
                "model": self.model_name,  
                "messages": messages,  
            }  

            if functions is not None:  
                payload["functions"] = functions  
                payload["function_call"] = function_call  

            completion = self.client.chat.completions.create(**payload)  
            message = completion.choices[0].message  

            if message.get("function_call"):  
                # 检测到模型调用 function  
                function_name = message.function_call.name  
                function_args_json = message.function_call.arguments  

                try:  
                    function_args = json.loads(function_args_json)  
                except Exception as e:  
                    raise ValueError(f"[ERROR] 解析函数参数失败: {e}\n内容: {function_args_json}")  

                if not available_functions or function_name not in available_functions:  
                    raise ValueError(f"[ERROR] function '{function_name}' 未在 available_functions 中注册。")  

                # 本地调用函数  
                py_func = available_functions[function_name]  
                try:  
                    function_response = py_func(**function_args)  
                except Exception as e:  
                    function_response = f"[Function Execution Error] {str(e)}"  

                # 记录调用步骤
                function_call_steps.append({
                    'function_name': function_name,
                    'arguments': function_args,
                    'response_from_function': function_response
                })

                # message 列表添加 function_call 记录
                messages.append(message)  # 系统发出的 function_call 消息
                messages.append({
                    'role': 'function',
                    'name': function_name,
                    'content': function_response
                })

                # 继续下一轮
                continue
            else:
                # 没有 function_call，返回 message
                return {
                    "final_message": message,
                    "all_messages": messages,
                    "function_call_steps": function_call_steps,
                }

        raise RuntimeError(f"[ERROR] 达到最大 {max_function_calls} 次连续函数调用，可能存在循环调用问题。")