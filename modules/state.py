import json
import copy

class SynthesisState():
    
    scenario: dict = None
    meta_data: str = None
    criteria: list[dict] = None

    message: list[dict] = None
    critique: list[str] = None
    scores: dict = None

    cost: float = 0.0

    def __init__(
        self,
        init_state: dict = None
    ) -> None:
        
        if init_state:
            self.set_state(init_state)

    def init_state(self) -> None:

        for key in self.__dict__.keys():
            setattr(self, key, None)
        self.cost = 0.0

    def set_state(self, state_dict: dict) -> None:

        for key, value in state_dict.items():
            setattr(self, key, value)

    def keys(self) -> set:

        return [
            key for key, value in self.__dict__.items()
            if value is not None
        ]
    
    def items(self) -> set:

        return [
            (key, value) for key, value in self.__dict__.items()
            if value is not None
        ]

    def to_dict(self, verbose: bool = False) -> dict:

        state_dict = {}
        if verbose:
            for key, value in self.items():
                if key == 'criteria':
                    criteria = copy.deepcopy(value)
                    for metric in criteria:
                        metric['levels'] = '<score levels>'
                    state_dict[key] = criteria
                else:
                    state_dict[key] = value
        else:
            for key, value in self.items():
                if key in ['message', 'cost']:
                    state_dict[key] = value
                elif key == 'scores':
                    state_dict[key] = {
                        model_name: {
                            score['criterion']: score['score']
                            for score in model_scores
                        } for model_name, model_scores in value.items()
                    }
                else:
                    state_dict[key] = f'<{key}>'
        
        return state_dict

    def to_str(self, verbose: bool = False) -> str:
        
        return json.dumps(self.to_dict(verbose), ensure_ascii = False)