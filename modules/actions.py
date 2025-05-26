from copy import deepcopy

from models import LLM
from modules.state import SynthesisState
from modules.prompt_templates import *
from modules.utils import *

def retry(
    max_attempt: int = 3,
):
    def decorator(func):
        def wrapper(*args, **kwargs):
            n_attempt = 0
            while n_attempt < max_attempt:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    n_attempt += 1
                    print(f"Attempt {n_attempt} failed: {e}")
                    if n_attempt == max_attempt:
                        raise e
        return wrapper
    return decorator

class Action():

    required_keys: tuple = ()
    description: str = ''

    def check_required_keys(
        self,
        state: SynthesisState,
        required_keys: tuple = None
    ):
        if required_keys == None:
            required_keys = self.required_keys
        
        for required_key in required_keys:
            if required_key.startswith('message'):
                last_role = required_key.split('_')[-1]
                if not state.message and last_role == 'assistant':
                    continue
                if state.message[-1]['role'] != last_role:
                    raise ValueError(f'Invalid message: the role of last message must be {last_role}.')
                
            elif required_key not in state.keys() or \
                getattr(state, required_key) is None:
                raise KeyError(f'Missing key \'{required_key}\' in state.')

    @property
    def parameters(self):

        properties = {}
        required = []
        for param_name, param_type in inspect_method(type(self), '__call__'):
            if param_type == None:
                continue
            elif param_type == SynthesisState:
                continue
            elif param_type == LLM:
                properties['model_name'] = {
                    'type': 'string',
                    'description': 'Name of LLM used to preform this action'
                }
            else:
                properties[param_name] = {
                    'type': str(param_type),
                    'description': ''
                }
            required.append(param_name)

        return {
            'type': 'object',  
            'properties': properties,
            'required': required
        }

class UserGenerate(Action):

    required_keys = ('scenario', 'meta_data', 'message_assistant')
    description = 'Play as user and send a query to assistant'

    def format_res(self, message: list, gen_res: str):

        json_obj = extract_json(gen_res)
        assert json_obj['role'] == 'user'

    @retry(max_attempt = 3)
    def __call__(
        self,
        state: SynthesisState,
        llm: LLM
    ) -> SynthesisState:
        self.check_required_keys(state)

        if state.message is None:
            system_prompt = system_template.format(
                task = state.scenario['task'],
                criteria = '\n'.join([c['metric'] for c in state.criteria])
            )
            state.message = [{'role': 'system', 'content': system_prompt}]
        
        prompt = user_generation_template.format(
            scenario = state.scenario,
            meta_data = state.meta_data,
            message = state.message
        )
        response = llm.get_response(
            messages = [{'role': 'user', 'content': prompt}, ]
        ).choices[0].message.content.strip()
        
        json_obj = extract_json(response)
        assert 'role' in json_obj and 'content' in json_obj
        assert json_obj['role'] == 'user'
        
        state.message.append({
            'role': 'user',
            'content': json_obj['content']
        })
        state.scores = None

        return state
    
class AssistantGenerate(Action):

    required_keys = ('message_user', )
    description = 'Play as assistant and response to user'

    @retry(max_attempt = 3)
    def __call__(
        self,
        state: SynthesisState,
        llm: LLM
    ) -> SynthesisState:
        self.check_required_keys(state)
        
        generation_res = llm.get_response(
            messages = state.message
        ).choices[0].message.content.strip()
        state.message.append({
            'role': 'assistant',
            'content': generation_res
        })
        state.scores = None
        
        return state
    
class Evaluate(Action):

    required_keys = ('scenario', 'message_assistant', 'criteria')
    description = 'Evaluate message with given criterias'

    def check_scores(self, scores: list, criteria: list):

        if set(score['criterion'] for score in scores) != \
            set(c['metric'] for c in criteria):
            raise ValueError(f'[Score Parse Error] Invalid criterion format: {scores}')
        
        for score in scores:
            value = score['score']
            if not isinstance(value, (int, float)):
                raise ValueError(f'[Score Parse Error] Invalid score value: {value}')

    @retry(max_attempt = 3)
    def __call__(
        self,
        state: SynthesisState,
        llm: LLM
    ) -> SynthesisState:
        self.check_required_keys(state)
        
        message = deepcopy(state.message)
        if message[0]['role'] == 'system':
            message = message[1:]

        prompt = evaluation_template.format(
            scenario = state.scenario,
            message = message,
            criteria = state.criteria
        )
        response = llm.get_response(
            messages = [{'role': 'user', 'content': prompt}, ],
            temperature = 0.0
        ).choices[0].message.content.strip()

        scores = extract_json(response)
        self.check_scores(scores, state.criteria)
        state.scores = scores
        
        return state
    
class Review(Action):

    required_keys = ('scenario', 'message_assistant', 'criteria')
    description = 'Review the message with given criterias'

    @retry(max_attempt = 3)
    def __call__(
        self,
        state: SynthesisState,
        llm: LLM
    ) -> SynthesisState:
        self.check_required_keys(state)

        message = deepcopy(state.message)
        if message[0]['role'] == 'system':
            message = message[1:]

        prompt = review_template.format(
            scenario = state.scenario,
            message = message,
            criteria = state.criteria
        )
        response = llm.get_response(
            messages = [{'role': 'user', 'content': prompt}, ]
        ).choices[0].message.content.strip()

        critique = extract_json(response)
        state.critique = critique
        
        return state
    
class Refine(Action):

    required_keys = ('scenario', 'message_assistant', 'critique')
    description = 'Refine message with critique'

    @retry(max_attempt = 3)
    def __call__(
        self,
        state: SynthesisState,
        llm: LLM
    ) -> SynthesisState:
        self.check_required_keys(state)

        message_dict = {
            str(idx): m for idx, m in enumerate(state.message)
            if m['role'] != 'system'
        }
        assistant_idxs = [
            idx for idx, m in message_dict.items()
            if m['role'] == 'assistant'
        ]

        prompt = refine_template.format(
            scenario = state.scenario,
            message = message_dict,
            assistant_idxs = assistant_idxs,
            critique = state.critique
        )
        response = llm.get_response(
            messages = [{'role': 'user', 'content': prompt}, ]
        ).choices[0].message.content.strip()
        
        refined_dict = extract_json(response)
        for idx, refined_m in refined_dict.items():
            if int(idx) not in assistant_idxs:
                continue
            assert refined_m['role'] == 'assistant'
            state.message[int(idx)]['content'] = refined_m['content']
        
        state.scores = None
        state.critique = None

        return state
    
class Output(Action):

    required_keys = ('message_assistant', )
    description = 'End the generation process and output result message'

    @retry(max_attempt = 3)
    def __call__(
        self,
        state: SynthesisState
    ) -> SynthesisState:
        self.check_required_keys(state)
        
        return state
    
