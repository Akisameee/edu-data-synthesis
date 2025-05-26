import json
from openai import OpenAI
from models.llm import LLM

class LLM_FunctionCalling():

    def __init__(self, llm: LLM) -> None:
        
        self.llm = llm
        self.model_name = llm.model_name
        self.client = llm.client

    def serialize_tool_call(self, tool_call):

        return {
            "id": tool_call.id,  
            "type": tool_call.type,  
            "function": {  
                "name": tool_call.function.name,  
                "arguments": tool_call.function.arguments,  
            }
        }
    
    def get_response(  
        self,  
        prompt: str,  
        tools: list = None,  
        available_tools: dict = None,  
        tool_choice = "auto",  
        memory_messages: list = None,  
        max_tool_calls: int = 5  
    ):  
        """  
        高级版获取回复，支持 tools 链式推理 + 多轮对话记忆 (新版 API)  
        
        Args:  
            prompt (str): 用户输入  
            tools (list, optional): 提供给模型的可调用工具（function 类型）  
            available_functions (dict, optional): 本地可直接执行 {'func_name': callable}  
            tool_choice (str/dict, optional): tool_choice 策略 'auto' / 'none' / {'function': {'name': xxx}}  
            memory_messages (list, optional): 已有对话历史  
            max_tool_calls (int): 最大允许连续 tool call 次数（防循环卡死）  

        Returns:  
            dict:  
                - "final_message": 最后生成内容  
                - "all_messages": 消息列表全过程  
                - "tool_call_steps": 中间的每步调用记录  
        """  

        if memory_messages is None:  
            memory_messages = []  

        messages = memory_messages + [{'role': 'user', 'content': prompt}]  
        tool_call_steps = []  

        for call_iteration in range(max_tool_calls + 1):  # 防御死循环  
            payload = {  
                "model": self.model_name,  
                "messages": messages,  
            }  

            if tools is not None:  
                payload["tools"] = tools  
                payload["tool_choice"] = tool_choice  

            completion = self.client.chat.completions.create(**payload)  
            message = completion.choices[0].message

            if hasattr(message, "tool_calls") and message.tool_calls:
                
                messages.append({
                    'role': 'assistant',  
                    'content': None,  
                    'tool_calls': [self.serialize_tool_call(tc) for tc in message.tool_calls]
                })

                for tool_call in message.tool_calls:  
                    tool_call_id = tool_call.id  
                    tool_name = tool_call.function.name  
                    function_args_json = tool_call.function.arguments

                    try:  
                        function_args = json.loads(function_args_json)  
                    except Exception as e:  
                        raise ValueError(f"[ERROR] 解析工具参数失败: {e}\n内容: {function_args_json}")  

                    if not available_tools or tool_name not in available_tools:  
                        raise ValueError(f"[ERROR] tool '{tool_name}' 未在 available_functions 中注册。")  

                    # 本地调用对应函数  
                    py_func = available_tools[tool_name]  
                    try:  
                        function_response = py_func(**function_args)  
                    except Exception as e:  
                        function_response = f"[Tool Execution Error] {str(e)}"  

                    # 记录调用步骤  
                    tool_call_steps.append({  
                        'tool_name': tool_name,  
                        'arguments': function_args,  
                        'response_from_tool': function_response,  
                    })  

                    # 将函数返回作为 function role 消息加入  
                    messages.append({  
                        'role': 'tool',       
                        'tool_call_id': tool_call_id,  
                        'name': tool_name,    
                        'content': function_response,  
                    })
                
                continue  

            else:  
                # 没有 tool_call，返回模型最终回答  
                return {  
                    "final_message": message.content,  
                    "all_messages": messages,  
                    "tool_call_steps": tool_call_steps,  
                }  

        raise RuntimeError(f"[ERROR] 达到最大 {max_tool_calls} 次连续 tool 调用，防止死循环。") 