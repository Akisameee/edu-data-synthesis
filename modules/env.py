import json
from openai.types.chat import ParsedFunctionToolCall,ChatCompletionMessageToolCall

from models import get_model
from modules.state import SynthesisState
from modules.sampler import Sampler
from modules.actions import *

[{
    "type": "function",
    "function": {
        "name": "sample_question",  
        "description": "根据学段（level）、学科（subject）、题目类型（type）从外部数据集随机采样问题数据，每条数据只会被采样一次",  
        "parameters": {  
            "type": "object",  
            "properties": {  
                "level": {
                    "type": "string",
                    "description": "学段，例如：primary, junior, senior, undergraduate, graduate, 传入None则不按此键过滤"
                },
                "subject": {
                    "type": "string",
                    "description": "科目，例如：chinese, math, english, 传入None则不按此键过滤"
                },
                "type_": {
                    "type": "string",
                    "description": "题目种类，例如：single_choice, fill_in_blank, 传入None则不按此键过滤"
                }
            },
            "required": []
        }
    }
}]

class Environment():

    def __init__(
        self,
        model_names: list = []
    ) -> None:
        
        self.state = SynthesisState()
        self.model_names = model_names
        self.llms = {model_name: get_model(model_name) for model_name in model_names}

        self.sampler = Sampler('./data/zh')
        self.user_generate = UserGenerate()
        self.assistant_generate = AssistantGenerate()
        self.evaluate = Evaluate()
        self.refine = Refine()
        self.output = Output()

        self.actions = {
            'sample_question': {
                'function': self._sample_question,
                'required_keys': []
            },
            'user_generate': {
                'function': self._user_generate,
                'required_keys': self.user_generate.required_keys
            },
            'assistant_generate': {
                'function': self._assistant_generate,
                'required_keys': self.assistant_generate.required_keys
            },
            'evaluate': {
                'function': self._evaluate,
                'required_keys': self.evaluate.required_keys
            },
            'refine': {
                'function': self._refine,
                'required_keys': self.refine.required_keys
            },
            'output': {
                'function': self._output,
                'required_keys': self.output.required_keys
            }
        }
        self.tools = [
            {
                'type': 'function',
                'function': {
                    'name': 'sample_question',
                    'description': '根据学段（level）、学科（subject）、题目类型（type）从外部数据集随机采样问题数据，每条数据只会被采样一次',
                    'parameters': {  
                        'type': 'object',  
                        'properties': {
                            'level': {
                                'type': 'string',
                                'description': '学段，例如：primary, junior, senior, undergraduate, graduate, 传入None则不按此键过滤'
                            },
                            'subject': {
                                'type': 'string',
                                'description': '科目，例如：chinese, math, english, 传入None则不按此键过滤'
                            },
                            'type_': {
                                'type': 'string',
                                'description': '题目种类，例如：single_choice, fill_in_blank, 传入None则不按此键过滤'
                            }
                        },
                        'required': []
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'user_generate',
                    'description': '为当前的message添加对话：作为user向assistant发出请求',
                    'parameters': {  
                        'type': 'object',  
                        'properties': {
                            'model_name': {
                                'type': 'string',
                                'description': '完成此过程使用的模型'
                            }
                        },
                        'required': []
                    }
                }
            },
        ]

    def init_env(self, init_state: dict):
        
        self.state.set_state(init_state)
        self.sampler.set_scope(init_state['theme']['task'])

        return self._get_stats()

    def _available_actions(self) -> list:

        available_actions = []
        for action, prop in self.actions.items():
            if all(key in self.state.keys() for key in prop['required_keys']):
                available_actions.append(action)

        return available_actions

    def _get_stats(self) -> str:

        state_info = f'state:\n{self.state.to_str()}'
        action_info = f'available_actions:\n{json.dumps(self._available_actions(), ensure_ascii = False)}'
        model_info = f'available_models:\n{json.dumps(self.model_names, ensure_ascii = False)}'

        return f'{state_info}\n{action_info}\n{model_info}\n'
    
    def _sample_question(self, **kwargs):

        self.state.meta_data = self.sampler.sample_question(**kwargs)

    def _user_generate(self, model_name: str):

        llm = self.llms[model_name]
        self.state = self.user_generate(self.state, llm)

    def _assistant_generate(self, model_name: str):

        llm = self.llms[model_name]
        self.state = self.assistant_generate(self.state, llm)

    def _evaluate(self, model_name: str):

        llm = self.llms[model_name]
        self.state = self.evaluate(self.state, llm)

    def _refine(self, model_name: str):

        llm = self.llms[model_name]
        self.state = self.refine(self.state, llm)

    def _output(self) -> str:

        self.state = self.output(self.state)

    def tool_call(self, tool_call: ChatCompletionMessageToolCall):
        
        try:
            tool_call_id = tool_call.id
            tool_name = tool_call.function.name
            if tool_name not in self.actions.keys():
                raise ValueError(f"[Unregistered Error] Tool \'{tool_name}\' unregistered.")
            
            function_args_json = tool_call.function.arguments
            try:
                function_args = json.loads(function_args_json)
            except Exception as e:
                raise ValueError("[Argument Error] Failed to parse arguments: {e}\nArgument content: {function_args_json}")
            
            py_func = self.actions[tool_name]['function']
            try:
                py_func(**function_args)
            except Exception as e:
                raise ValueError(f"[Tool Execution Error] {str(e)}")
            
            response = self._get_stats()
        
        except Exception as e:
            response = str(e)

        return {
            'role': 'tool',
            'tool_call_id': tool_call_id,
            'name': tool_name,
            'content': response,
        }
