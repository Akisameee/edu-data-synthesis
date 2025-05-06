from modules.action_node import ActionNode
from models.llm import LLM
from modules.utils import extract_json

evaluation_template = \
'''
我将向你提供一段教育领域下特定场景的对话，请根据所给定的所有评估指标及其评分细则对所给的回答进行评分并给出理由。
以json的格式返回，例如：
```json[{{'criterion': <评估指标1名称>, 'score': <得分>, 'reason': <理由>}}, {{'criterion': <评估指标2名称>, ...}}, ...]```

场景：
{theme}
对话：
{message}
评估指标: 
{criteria}
'''

class EvaluateNode(ActionNode):

    required_keys = ['theme', 'message', 'criteria']

    def __call__(
        self,
        state: dict,
        llm: LLM
    ) -> dict:
        super().__call__(state)

        theme, message, criteria = (
            state['theme'], state['message'], state['criteria']
        )
        if message[-1]['role'] != 'assistant':
            raise ValueError(f'Incomplete message: the role of last message must be assistant.')
        
        prompt = evaluation_template.format(
            theme = theme,
            message = message,
            criteria = criteria
        )
        response = llm.get_response(
            message = [{'role': 'user', 'content': prompt}, ]
        )

        scores = extract_json(response)
        state.setdefault('scores', []).append(scores)
        
        return state