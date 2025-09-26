from typing import List, Literal, Optional
from dataclasses import dataclass, field
from langchain_community.tools import BaseTool, TavilySearchResults

import sys
sys.path.insert(0, '..')

from modules.utils import get_config_value

@dataclass
class WebSearchConfig:
    engine: Literal['tavily'] = get_config_value('websearch.engine')
    api_key: str = get_config_value('websearch.api_key')
    max_search_results: int = 5
    include_domains: Optional[List[str]] = field(default_factory=list)
    exclude_domains: Optional[List[str]] = field(default_factory=list)

def get_web_search_tool(config: WebSearchConfig) -> BaseTool:
    return TavilySearchResults(
        name = 'web_search',
        tavily_api_key = config.api_key,
        max_results = config.max_search_results,
        include_raw_content = False,
        include_answer = True,
        include_domains = config.include_domains,
        exclude_domains = config.exclude_domains
    )

if __name__ == '__main__':
    tool = get_web_search_tool(WebSearchConfig())
    res = tool.invoke('langchain')
    print(res)

