import json

from models.llm import Base_LLM, LLM_API
from models.function_call import LLM_FunctionCalling

with open('./models/api_keys.json', 'r') as file:
    api_keys = json.load(file)

model_map = {
    'qwen2.5-7b-instruct': {
        'class': LLM_API,
        'kwargs': {
            'model_name': 'Qwen/Qwen2.5-7B-Instruct',
            **api_keys['siliconflow'],
            'price': {
                'prompt': 0.00035 / 1000,
                'completion': 0.00035 / 1000
            }
        }
    },
    'qwen2.5-14b-instruct': {
        'class': LLM_API,
        'kwargs': {
            'model_name': 'Qwen/Qwen2.5-14B-Instruct',
            **api_keys['siliconflow'],
            'price': {
                'prompt': 0.0007 / 1000,
                'completion': 0.0007 / 1000
            }
        }
    },
    'qwen-max': {
        'class': LLM_API,
        'kwargs': {
            'model_name': 'qwen-max',
            **api_keys['aliyuncs'],
            'price': {
                'prompt': 0.0024 / 1000,
                'completion': 0.0096 / 1000
            }
        }
    },
    # 'deepseek-v3': {
    #     'class': LLM_API,
    #     'kwargs': {
    #         'model_name': 'Pro/deepseek-ai/DeepSeek-V3',
    #         **api_keys['siliconflow']
    #     }
    # },
    # 'deepseek-r1': {
    #     'class': LLM_API,
    #     'kwargs': {
    #         'model_name': 'Pro/deepseek-ai/DeepSeek-R1',
    #         **api_keys['siliconflow']
    #     }
    # },
    'deepseek-v3': {
        'class': LLM_API,
        'kwargs': {
            'model_name': 'deepseek-v3',
            **api_keys['chatanywhere'],
            'price': {
                'prompt': 0.0012 / 1000,
                'completion': 0.0048 / 1000
            }
        }
    },
    'deepseek-r1': {
        'class': LLM_API,
        'kwargs': {
            'model_name': 'deepseek-r1',
            **api_keys['chatanywhere'],
            'price': {
                'prompt': 0.0024 / 1000,
                'completion': 0.0096 / 1000
            }
        }
    },
    'gpt-4o': {
        'class': LLM_API,
        'kwargs': {
            'model_name': 'gpt-4o',
            **api_keys['chatanywhere'],
            'price': {
                'prompt': 0.0175 / 1000,
                'completion': 0.07 / 1000
            }
        }
    },
}

def get_model(
    model_name: str
) -> Base_LLM:
    
    model = model_map[model_name]['class'](
        **model_map[model_name]['kwargs']
    )
    
    return model