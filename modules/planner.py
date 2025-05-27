import json
from tqdm import tqdm
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function

from modules.utils import extract_json
from modules.prompt_templates import planning_template
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

    def run_function_calling(
        self,
        init_state: dict,
        max_tool_call: int = 10
    ):
        trajectory = []
        init_response = self.env.init_env(init_state)
        # print(init_response)

        planning_agent = get_model('deepseek-v3')
        planning_prompt = planning_template.format(
            task = self.env.state.scenario['task'],
            init_state = init_response
        )
        trajectory = [{'role': 'user', 'content': planning_prompt}, ]

        for n_tool_call in range(max_tool_call):
            completion = planning_agent.get_response(
                messages = trajectory,
                tools = self.env.tools,
                tool_choice = 'auto',
            )
            message = completion.choices[0].message
            trajectory.append({
                'role': 'assistant',
                'content': message.content.strip() if message.content else None,
                'tool_calls': message.tool_calls
            })

            if message.tool_calls:
                tool_call = message.tool_calls[0]
                tool_call_message = self.env.tool_call(tool_call)

                raised_error = tool_call_message.pop('error')
                if raised_error:
                    tqdm.write(tool_call_message['content'])
                trajectory.append(tool_call_message)
                
                if tool_call_message['name'] == 'output':
                    break

        for action in trajectory:
            if action['role'] == 'tool':
                action['content'] = json.loads(action['content'])
        
        return self.env.state, trajectory
        
    def run_seq_workflow(
        self,
        init_state: dict,
        workflow: list
    ):
        trajectory = []
        init_response = self.env.init_env(init_state)
        # print(init_response)
        for tool_call_args in workflow:
            state_dict = self.env.state.to_dict()
            tool_call_message = self.tool_call(
                *tool_call_args
            )
            if tool_call_message['error']:
                raise RuntimeError(tool_call_message['content'])
            # print(tool_call_message['content'])
            trajectory.append(tool_call_message)

        return self.env.state, trajectory


