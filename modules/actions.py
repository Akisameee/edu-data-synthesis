from models import LLM
from modules.state import SynthesisState
from modules.prompt_templates import *
from modules.utils import extract_json

class Action():

    required_keys = []

    def __call__(self, state: SynthesisState):
        
        for required_key in self.required_keys:
            if required_key not in state.keys() or \
                getattr(state, required_key) is None:
                raise KeyError(f'Missing key \'{required_key}\' in state.')

class UserGenerate(Action):

    required_keys = ['theme', 'meta_data']

    def format_res(self, message: list, gen_res: str):

        json_obj = extract_json(gen_res)
        assert json_obj['role'] == 'user'

    def __call__(
        self,
        state: SynthesisState,
        llm: LLM
    ):
        super().__call__(state)

        message = state.message_response if state.message_response is not None else []
        if len(message) > 0 and message[-1]['role'] != 'assistant':
            raise ValueError(f'Invaild query sequence.')
        
        prompt = user_generation_template.format(
            theme = state.theme,
            meta_data = state.meta_data,
            message = message
        )
        response = llm.get_response(
            message = [{'role': 'user', 'content': prompt}, ]
        )
        
        json_obj = extract_json(response)
        assert 'role' in json_obj and 'content' in json_obj
        assert json_obj['role'] == 'user'
        
        message.append({'role': 'user', 'content': json_obj['content']})
        state.message_query = message
        state.message_response = None
        state.scores = None

        return state
    
class AssistantGenerate(Action):

    required_keys = ['message_query']

    def __call__(
        self,
        state: SynthesisState,
        llm: LLM
    ) -> dict:
        super().__call__(state)

        message = state.message_query
        if message[-1]['role'] != 'user':
            raise ValueError(f'Invaild query sequence.')
        
        generation_res = llm.get_response(
            message = message
        )
        message.append({
            'role': 'assistant',
            'content': generation_res
        })
        state.message_response = message
        state.message_query = None
        state.scores = None
        
        return state
    
class Evaluate(Action):

    required_keys = ['theme', 'message_response', 'criteria']

    def __call__(
        self,
        state: SynthesisState,
        llm: LLM
    ) -> dict:
        super().__call__(state)

        message = state.message_response
        if message[-1]['role'] != 'assistant':
            raise ValueError(f'Incomplete message: the role of last message must be assistant.')
        
        prompt = evaluation_template.format(
            theme = state.theme,
            message = message,
            criteria = state.criteria
        )
        response = llm.get_response(
            message = [{'role': 'user', 'content': prompt}, ]
        )

        scores = extract_json(response)

        scores_all = state.scores if state.scores else []
        scores_all.append(scores)
        state.scores = scores_all
        
        return state
    
class Refine(Action):

    required_keys = ['theme', 'message_response', 'scores']

    def __call__(
        self,
        state: SynthesisState,
        llm: LLM
    ) -> dict:
        super().__call__(state)

        message = state.message_response
        if message[-1]['role'] != 'assistant':
            raise ValueError(f'Incomplete message: the role of last message must be assistant.')
        
        # scores = [
        #     [s for s in score if int(s['score']) >= 9]
        #     for score in scores
        # ]
        prompt = refine_template.format(
            theme = state.theme,
            message = message,
            scores = state.scores
        )
        response = llm.get_response(
            message = [{'role': 'user', 'content': prompt}, ]
        )
        
        refine_res = extract_json(response)
        state.message_response = refine_res
        state.scores = None

        return state
    
class Output(Action):

    required_keys = ['message_response']

    def __call__(
        self,
        state: SynthesisState
    ) -> dict:
        super().__call__(state)

        message = state.message_response
        if message[-1]['role'] != 'assistant':
            raise ValueError(f'Incomplete message: the role of last message must be assistant.')
        
        return state
    
