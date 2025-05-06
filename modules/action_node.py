class ActionNode():

    required_keys = []

    def __call__(self, state: dict):
        
        for required_key in self.required_keys:
            if required_key not in state.keys():
                raise KeyError(f'Missing key \'{required_key}\' in state.')