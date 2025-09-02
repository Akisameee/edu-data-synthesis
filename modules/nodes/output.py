import sys
sys.path.insert(0, '..')

from modules.nodes.base import *
from modules.nodes.prompt_templates import *

class Output(Node):
    input_type = AssistantMessages | EvalScores
    output_type = AssistantMessages | EvalScores
    max_indegree = 1

    def __init__(self) -> None:
        super().__init__(None)

    async def __call__(
        self,
        output: AssistantMessages | EvalScores,
        **kwargs
    ) -> AssistantMessages | EvalScores:
        return output

class GenerationOutput(Output):
    input_type = AssistantMessages
    output_type = AssistantMessages

class EvaluationOutput(Output):
    input_type = EvalScores
    output_type = EvalScores