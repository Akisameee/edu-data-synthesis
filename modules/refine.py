from modules.action_node import ActionNode
from models.llm import LLM
from modules.utils import extract_json

refine_template = \
'''
我将向你提供一段教育领域下特定场景的对话以及其在几个评估指标下的得分和原因，请根据得分对原对话中assistant的回应进行改进（仅修改assistant的content）。
以json的对话格式返回，例如：
```json[{{'role': 'user', 'content': '...'}}, {{'role': 'assistant', 'content': '...'}}, ...]```

场景：
{theme}
对话：
{message}
得分: 
{scores}
'''

class RefineNode(ActionNode):

    required_keys = ['theme', 'message', 'scores']

    def __call__(
        self,
        state: dict,
        llm: LLM
    ) -> dict:
        super().__call__(state)

        theme, message, scores = (
            state['theme'], state['message'], state['scores']
        )
        if message[-1]['role'] != 'assistant':
            raise ValueError(f'Incomplete message: the role of last message must be assistant.')
        
        scores = [
            [s for s in score if int(s['score']) >= 9]
            for score in scores
        ]
        prompt = refine_template.format(
            theme = theme,
            message = message,
            scores = scores
        )
        response = llm.get_response(
            message = [{'role': 'user', 'content': prompt}, ]
        )

        refine_res = extract_json(response)
        state['message'] = refine_res
        del state['scores']

        return state