import sys
sys.path.insert(0, '..')

from modules.nodes.base import *
from modules.nodes.prompt_templates import *

class Identity(Node):
    input_state = Any
    output_state = input_state

    async def __call__(self, messages: Messages) -> Messages:
        return messages
    
class Input(Identity):
    max_indegree = 0

class GenerationInput(Input):
    input_state = 'user'
    output_state = 'user'

class EvaluationInput(Input):
    input_state = 'assistant'
    output_state = 'assistant'
    
class Output(Identity):
    max_indegree = 1
    max_outdegree = 0

class GenerationOutput(Output):
    input_state = 'assistant'
    output_state = 'assistant'

class EvaluationOutput(Output):
    input_state = 'scored'
    output_state = 'scored'