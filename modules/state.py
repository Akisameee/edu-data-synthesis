import json
import copy

class SynthesisState():
    
    theme: dict = None
    meta_data = None
    criteria: list[dict] = None

    message_query: list[dict] = None
    message_response: list[dict] = None
    scores: list[dict] = None

    def __init__(
        self,
        init_state: dict = None
    ) -> None:
        
        if init_state:
            self.set_state(init_state)

    def set_state(self, state_dict: dict) -> None:

        for key, value in state_dict.items():
            setattr(self, key, value)

    def keys(self) -> set:

        return self.__dict__.keys()

    def to_dict(self) -> dict:

        state_dict = {}
        for key, value in self.__dict__.items():
            if not value:
                continue
            if key == 'criteria':
                criteria = copy.deepcopy(value)
                for metric in criteria:
                    metric['levels'] = '<score levels>'
                state_dict[key] = criteria
            else:
                state_dict[key] = value
        
        return state_dict

    def to_str(self) -> str:
        
        return json.dumps(self.to_dict(), ensure_ascii = False)