from modules.action_node import ActionNode
from models.llm import LLM

user_generation_template = \
'''
你是一个教育领域大模型的用户，请使用元数据/历史数据模拟用户在给定的教育场景下向大模型发出一次询问/对话。
以json的对话格式返回，例如：
```json[{{'role': 'user','content': '...'}}]```

场景:
{theme}
元数据:
{meta_data}
对话历史：
{message}
'''

class UserGenerateNode(ActionNode):

    required_keys = ['theme', 'meta_data']

    def __call__(
        self,
        state: dict,
        llm: LLM
    ):
        theme, meta_data, message = (
            state['theme'], state['meta_data'],
            state['message'] if 'message' in state.keys() else []
        )

        if len(message) > 0 and message[-1]['role'] != 'assistant':
            raise ValueError(f'Invaild query sequence.')
        
        prompt = user_generation_template.format(
            theme = theme,
            meta_data = meta_data,
            message = message
        )
        generation_res = llm.get_response(
            message = [{'role': 'user', 'content': prompt}, ]
        )

        state['message'] = message.append({'role': 'user', 'content': generation_res})
        return state