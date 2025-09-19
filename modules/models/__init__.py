import json

from modules.models.llm import Base_LLM, LLM_API, RM_HF
from modules.models.function_call import LLM_FunctionCalling

with open('./modules/models/api_keys.json', 'r') as file:
    api_keys = json.load(file)

model_map = {
    'qwen2.5-7b-instruct': {
        'class': LLM_API,
        'model_name_client': 'Qwen/Qwen2.5-7B-Instruct',
        **api_keys['siliconflow'],
        'price': {
            'prompt': 0.00035 / 1000,
            'completion': 0.00035 / 1000
        }
    },
    'qwen2.5-14b-instruct': {
        'class': LLM_API,
        'model_name_client': 'Qwen/Qwen2.5-14B-Instruct',
        **api_keys['siliconflow'],
        'price': {
            'prompt': 0.0007 / 1000,
            'completion': 0.0007 / 1000
        }
    },
    'qwen-max': {
        'class': LLM_API,
        'model_name_client': 'qwen-max',
        **api_keys['aliyuncs'],
        'price': {
            'prompt': 0.0024 / 1000,
            'completion': 0.0096 / 1000
        }
    },
    # 'deepseek-v3': {
    #     'class': LLM_API,
    #     'model_name_client': 'deepseek-chat',
    #     **api_keys['deepseek'],
    #     'price': {
    #         'prompt': 4e-6,
    #         'completion': 1.2e-5
    #     },
    # },
    # 'deepseek-r1': {
    #     'class': LLM_API,
    #     'model_name_client': 'deepseek-reasoner',
    #     **api_keys['deepseek'],
    #     'price': {
    #         'prompt': 4e-6,
    #         'completion': 1.2e-5
    #     }
    # },
    'deepseek-v3': {
        'class': LLM_API,
        'model_name_client': 'deepseek-v3.1-250821',
        **api_keys['chatanywhere'],
        'price': {
            'prompt': 0.0024 / 1000,
            'completion': 0.0072 / 1000
        }
    },
    'deepseek-r1': {
        'class': LLM_API,
        'model_name_client': 'deepseek-v3.1-think-250821',
        **api_keys['chatanywhere'],
        'price': {
            'prompt': 0.0024 / 1000,
            'completion': 0.0072 / 1000
        }
    },
    'gpt-4o': {
        'class': LLM_API,
        'model_name_client': 'gpt-4o',
        **api_keys['chatanywhere'],
        'price': {
            'prompt': 0.0175 / 1000,
            'completion': 0.07 / 1000
        }
    },
    'Skywork-Reward-V2-Llama-3.1-8B': {
        'class': RM_HF,
        'model_path': '/home/smliu/huggingface/Skywork/Skywork-Reward-V2-Llama-3.1-8B',
        'temperature': 0.0
    }
}

def get_model(
    model_name: str
) -> Base_LLM:
    
    kwargs = model_map[model_name].copy()
    model_cls = kwargs.pop('class')
    model = model_cls(
        model_name = model_name,
        **kwargs
    )
    
    return model