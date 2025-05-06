import json
from modules.user_generate import GeneratorAgent

class Environment():

    def __init__(
        self,
        model_names: list = []
    ) -> None:
        
        self.state = {}
        self.model_names = []

    def available_actions(self) -> list:

        pass

    def _get_stats(self) -> str:

        state_info = f'state:\n{json.dumps(self.state)}\n'

    def get_meta_data(self) -> str:

        pass

    def user_generate(self, model_name: str) -> str:

        pass

    def assistant_generate(self, model_name: str) -> str:

        pass

    def evaluate(self, model_name: str) -> str:

        pass

    def refine(self, model_name: str) -> str:
        
        pass

    def output(self) -> str:

        pass