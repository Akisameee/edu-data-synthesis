import logging
import os
from typing import List, Literal, Annotated, Optional
from langchain_community.tools import BaseTool

from modules.tools.web_search import WebSearchConfig, get_web_search_tool
from modules.tools.python_repl import python_repl_tool

supported_tools = [
    'web_search',
    'python_repl'
]

def get_tools(tools: List[str]) -> List[BaseTool]:
    langchain_tools = []
    for tool in tools:
        if tool == 'web_search':
            config = WebSearchConfig()
            langchain_tools.append(get_web_search_tool(config))
        elif tool == 'python_repl':
            langchain_tools.append(python_repl_tool)
        else:
            raise NotImplementedError
    return langchain_tools