from modules.action_node import ActionNode
from models.llm import LLM

class AssistantGenerateNode(ActionNode):

    required_keys = ['message']

    def __call__(
        self,
        state: dict,
        llm: LLM
    ):
        super.__call__(state)

        message = state['message']
        if message[-1]['role'] != 'user':
            raise ValueError(f'Invaild query sequence.')
        
        generation_res = llm.get_response(
            message = message
        )

        return generation_res