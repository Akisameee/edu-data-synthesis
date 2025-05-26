import json

from models.llm import LLM, LLM_API
from models.function_call import LLM_FunctionCalling

with open('./models/api_keys.json', 'r') as file:
    api_keys = json.load(file)

model_map = {
    'qwen2.5-7b-instruct': {
        'class': LLM_API,
        'kwargs': {
            'model_name': 'Qwen/Qwen2.5-7B-Instruct',
            **api_keys['siliconflow']
        }
    },
    'qwen2.5-14b-instruct': {
        'class': LLM_API,
        'kwargs': {
            'model_name': 'Qwen/Qwen2.5-14B-Instruct',
            **api_keys['siliconflow']
        }
    },
    'qwen-max': {
        'class': LLM_API,
        'kwargs': {
            'model_name': 'qwen-max',
            **api_keys['aliyuncs']
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
            **api_keys['chatanywhere']
        }
    },
    'deepseek-r1': {
        'class': LLM_API,
        'kwargs': {
            'model_name': 'deepseek-r1',
            **api_keys['chatanywhere']
        }
    },
    'gpt-4o': {
        'class': LLM_API,
        'kwargs': {
            'model_name': 'gpt-4o',
            **api_keys['chatanywhere']
        }
    },
}

def get_model(
    model_name: str
) -> LLM:
    
    model = model_map[model_name]['class'](
        **model_map[model_name]['kwargs']
    )
    
    return model