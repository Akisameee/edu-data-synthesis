import json
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function

from modules.utils import extract_json
from models import get_model
from modules.env import Environment

class Planner():

    def __init__(
        self,
        model_names: str,
    ) -> None:
        
        self.env = Environment(model_names)

    def tool_call(
        self,
        id: str,
        tool_name: str,
        arguments: dict
    ):
        func = Function(
            name = tool_name, arguments = json.dumps(arguments)
        )
        tool_call = ChatCompletionMessageToolCall(
            id = id, function = func, type = 'function'
        )
        return self.env.tool_call(tool_call)

    def run(
        self,
        init_state: dict
    ):
        trajectory = []
        init_response = self.env.init_env(init_state)
        print(init_response)

        workflow = [
            ('-1', 'sample_question', {'level': 'junior', 'subject': 'math'}),
            ('0', 'user_generate', {'model_name': 'deepseek-v3'}),
            ('1', 'assistant_generate', {'model_name': 'deepseek-v3'}),
            ('2', 'evaluate', {'model_name': 'deepseek-r1'}),
            ('3', 'refine', {'model_name': 'deepseek-v3'}),
            ('4', 'output', {})
        ]
        for tool_call_args in workflow:
            state_dict = self.env.state.to_dict()
            tool_call_message = self.tool_call(
                *tool_call_args
            )
            print(tool_call_message['content'])
            trajectory.append({
                'state': state_dict,
                'action': tool_call_message
            })

        return self.env.state.message_response, trajectory

