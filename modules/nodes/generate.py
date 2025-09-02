import sys
sys.path.insert(0, '..')

from modules.nodes.base import *
from modules.nodes.prompt_templates import *

class SystemGenerate(Node):
    output_type = SystemMessages

    def __init__(self, llm: Base_LLM = None) -> None:
        super().__init__(llm)

    async def __call__(
        self,
        **kwargs
    ) -> SystemMessages:
        
        system_prompt = system_template.format(
            task = kwargs['scenario']['task'],
            criteria = '\n'.join([c['metric'] for c in kwargs['criteria']])
        )
        return Messages([Message(role = 'system', content = system_prompt)])

class UserGenerate(Node):
    input_type = (SystemMessages, AssistantMessages)
    output_type = UserMessages

    def __init__(self, llm: Base_LLM = None) -> None:
        super().__init__(llm)

    @staticmethod
    def replace_meta_data(content: str, meta_data: str):

        if '[meta_data]' in content:
            content = content.replace('[meta_data]', meta_data, 1)
        else:
            content = f'{meta_data}\n{content}'
        return content

    @retry(max_attempt = 3)
    async def __call__(
        self,
        messages: Messages,
        **kwargs
    ) -> UserMessages:
        
        prompt = user_generate_template.format(
            scenario = kwargs['scenario'],
            meta_data = kwargs['meta_data'],
            message = messages.to_json()
        )

        completion = await self.llm.get_response(
            messages = [{'role': 'user', 'content': prompt}, ]
        )
        self.llm.cost(completion)
        response = completion.choices[0].message.content.strip()
        
        json_obj = extract_json(response)
        assert json_obj['role'] == 'user'
        assert 'role' in json_obj and 'content' in json_obj
        assert json_obj['role'] == 'user'
        
        messages.append(Message(
            role = 'user',
            content = self.replace_meta_data(
                json_obj['content'], kwargs['meta_data']
            )
        ))
        return messages
    
class AssistantGenerate(Node):
    input_type = UserMessages
    output_type = AssistantMessages

    def __init__(self, llm: Base_LLM = None) -> None:
        super().__init__(llm)

    @retry(max_attempt = 3)
    async def __call__(
        self,
        messages: UserMessages,
        **kwargs
    ) -> AssistantMessages:
        
        completion = await self.llm.get_response(messages = messages.to_json())
        self.llm.cost(completion)
        response = completion.choices[0].message.content.strip()

        messages.append(Message(
            role = 'assistant',
            content = response
        ))
        return messages
    
class ResponseAggregate(Node):
    input_type = List[AssistantMessages]
    output_type = AssistantMessages

    def __init__(self, llm: Base_LLM = None) -> None:
        super().__init__(llm)

    @retry(max_attempt = 3)
    async def __call__(
        self,
        messages_list: List[AssistantMessages],
        **kwargs
    ) -> AssistantMessages:
        n_messages = len(messages_list)
        if n_messages == 1:
            return messages_list[0]
        
        for i in range(n_messages):
            if any(msg != messages_list[i][0] for msg in messages_list[i]):
                break

        history = messages_list[0].to_json()[:i]
        responses = [messages[-1] for messages in messages_list]

        prompt = response_aggregate_template.format(
            scenario = kwargs['scenario'],
            history = history
        ) + '\n' + ''.join([f'Response {idx}:\n{response}\n' for idx, response in enumerate(responses)])
        
        completion = await self.llm.get_response(
            messages = [{'role': 'user', 'content': prompt}, ]
        )
        self.llm.cost(completion)
        response = completion.choices[0].message.content.strip()

        json_obj = extract_json(response)
        assert 'role' in json_obj and 'content' in json_obj
        assert json_obj['role'] == 'assistant'

        history.append(Message(
            role = 'assistant',
            content = json_obj['content']
        ))
        return history