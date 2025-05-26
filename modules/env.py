import json
from openai.types.chat import ChatCompletionMessageToolCall

from models import get_model
from modules.state import SynthesisState
from modules.sampler import SampleQuestion
from modules.actions import *

sample_question_tool = {
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
}

class Environment():

    def __init__(
        self,
        model_names: list = []
    ) -> None:
        
        self.state = SynthesisState()
        self.model_names = model_names
        self.llms = {model_name: get_model(model_name) for model_name in model_names}

        # self.sample_question = SampleQuestion('./data/zh')
        self.user_generate = UserGenerate()
        self.assistant_generate = AssistantGenerate()
        self.evaluate = Evaluate()
        self.review = Review()
        self.refine = Refine()
        self.output = Output()
    
    @property
    def _actions(self) -> dict:

        return {
            attr: getattr(self, attr) for attr in self.__dict__.keys()
            if not attr.startswith('_') and isinstance(getattr(self, attr), Action)
        }
    
    @property
    def _available_actions(self) -> list:

        available_actions = []
        for action_name, action_obj in self._actions.items():
            try:
                action_obj.check_required_keys(self.state)
                available_actions.append(action_name)
            except:
                continue

        return available_actions

    @property
    def tools(self):

        tools = []
        for action_name, action_obj in self._actions.items():
            tool = {
                'type': 'function',
                'function': {
                    'name': action_name,
                    'description': action_obj.description,
                    'parameters': action_obj.parameters
                }
            }
            tools.append(tool)
        
        return tools
    
    @property
    def functions(self):
        
        return {
            action_name: getattr(self, f'_{action_name}')
            for action_name in self._actions.keys()
        }

    def init_env(self, init_state: dict):
        
        self.state.init_state()
        self.state.set_state(init_state)
        self.sample_question.set_scope(init_state['scenario']['task'])

        return self._get_stats()

    def _get_stats(self) -> str:

        state_info = f'state:\n{self.state.to_str()}'
        action_info = f'available_actions:\n{json.dumps(self._available_actions, ensure_ascii = False)}'
        model_info = f'available_models:\n{json.dumps(self.model_names, ensure_ascii = False)}'

        return f'{state_info}\n{action_info}\n{model_info}\n'
    
    def _sample_question(self, **kwargs):

        self.state.meta_data = self.sample_question(**kwargs)

    def _user_generate(self, model_name: str):

        llm = self.llms[model_name]
        self.state = self.user_generate(self.state, llm)

    def _assistant_generate(self, model_name: str):

        llm = self.llms[model_name]
        self.state = self.assistant_generate(self.state, llm)

    def _evaluate(self, model_name: str):

        llm = self.llms[model_name]
        self.state = self.evaluate(self.state, llm)

    def _review(self, model_name: str):

        llm = self.llms[model_name]
        self.state = self.review(self.state, llm)

    def _refine(self, model_name: str):

        llm = self.llms[model_name]
        self.state = self.refine(self.state, llm)

    def _output(self) -> str:

        self.state = self.output(self.state)

    def tool_call(self, tool_call: ChatCompletionMessageToolCall):
        
        raised_error = False
        try:
            tool_call_id = tool_call.id
            tool_name = tool_call.function.name
            if tool_name not in self._actions.keys():
                raise ValueError(f'[Unregistered Error] Tool \'{tool_name}\' unregistered.')
            
            function_args_json = tool_call.function.arguments
            try:
                function_args = json.loads(function_args_json)
            except Exception as e:
                raise ValueError('[Argument Error] Failed to parse arguments: {e}\nArgument content: {function_args_json}')
            
            py_func = self.functions[tool_name]
            try:
                py_func(**function_args)
            except Exception as e:
                raise ValueError(f'[Tool Execution Error] {str(e)}')
            
            response = self._get_stats()
        
        except Exception as e:
            response = str(e)
            raised_error = True

        return {
            'role': 'tool',
            'tool_call_id': tool_call_id,
            'name': tool_name,
            'content': response,
            'error': raised_error
        }
