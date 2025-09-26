import json

from modules.utils import get_config_value
from modules.models.llm import Base_LLM, LLM_API, RM_HF

model_configs = get_config_value('models')
provider_configs = get_config_value('providers')

def get_model(
    name: str
) -> Base_LLM:
    if name not in model_configs:
        raise ValueError(f'Model {name} is not registered in config.yaml')
    model_config: dict = model_configs[name]
    provider = model_config.pop('provider')
    if provider not in provider_configs:
        raise ValueError(f'Provider {provider} is not registered in config.yaml')
    provider_config: dict = provider_configs[provider]
    model_config.update(provider_config)
    model = LLM_API(name = name, **model_config)
    return model