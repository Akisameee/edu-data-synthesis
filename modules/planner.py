import json

from modules.utils import extract_json
from models.llm import LLM, LLMFunctionCalling
from modules.sampler import SamplerModule

planning_template = \
'''
你是一个规划智能体，你需要规划教育领域的场景数据生成过程，并通过调用函数执行整个过程。
生成的数据为json格式的对话，例如：
```json[{{'role': 'user','content': '...'}},{{'role': 'assistant','content': '...'}}]```
其中user是教育领域大模型的主要服务对象，如学生、教师等；assistant需要尽可能地满足服务对象在特定教育场景下给出的指令。

大致的生成过程为：
1. 获得元数据：为了保证数据的多样性，减少重复，每一条数据都尽量需要使用元数据生成
    -对于和具体题目相关的场景，可从外部数据库中采样题目作为元数据
        -在进行采样前，可以通过相关方法查看外部数据库有哪些内容
        -采样得到的数据有些有答案和解析过程
    -对于和学生画像相关的场景，可以生成元数据并进行保存，生成的时候再去采样
2. 生成数据：生成提示并使用提示获取assistant模型的响应
    -**在user的视角**使用给定的场景和元数据生成一个提示，确保被询问的模型是辅助教学任务的那一方
    -确保在提示中使用了meta_data中的内容，根据场景不同考虑user是否会得知标准答案和解析，如果不会不要把相关内容写进提示内
    -调用相关方法获取assistant模型的响应
3. 整合数据：将第2步生成的提示和获得的模型响应整理成目标json格式

注意：
    -生成面对**中文场景**的教育场景数据，确保对话双方的主体语言为中文（外语科目会不可避免地涉及到外语，但也请确保对话的主题语言为中文）

教学场景/任务：
{task}
'''

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_question_database_info",
            "description": "获得外部题目数据库的信息，返回所有的学段、学科、题目类型以及其剩余的样本数量",
            "parameters": {
                "type": "object",
                "properties": {},
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sample_question",  
            "description": "根据学段（level）、学科（subject）、题目类型（type）从外部数据集随机采样问题数据，每条数据只会被采样一次",  
            "parameters": {  
                "type": "object",  
                "properties": {  
                    "level": {
                        "type": "string",
                        "description": "学段，例如：primary, junior, senior, undergraduate, graduate, 传入None则不按此键过滤"
                    },
                    "subject": {
                        "type": "string",
                        "description": "科目，例如：chinese, math, english, 传入None则不按此键过滤"
                    },
                    "type_": {
                        "type": "string",
                        "description": "题目种类，例如：single_choice, fill_in_blank, 传入None则不按此键过滤"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_response",
            "description": "使用提示词（prompt）生成模型的响应",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "生成数据的指令"
                    }
                },
                "required": ["prompt"]
            }
        }
    },
]

class PlannerAgent():

    def __init__(
        self,
        model_name: str,
    ) -> None:
        
        self.planner = LLMFunctionCalling(model_name)
        self.sampler = SamplerModule('./data/zh', scope = None)
        self.generator = generator
        self.available_tools = {
            'get_question_database_info': self.sampler.get_question_database_info,
            'sample_question': self.sampler.sample_question,
            'generate_response': self.generator.generate_response
        }

    def set_scope(self, scope):

        self.sampler.set_scope(scope)

    def run(
        self,
        scope: str,
        task: str
    ):
        self.set_scope(scope)
        
        planning_prompt = planning_template.format(
            task = task
        )
        res = self.planner.get_response(
            prompt = planning_prompt,
            tools = tools,
            available_tools = self.available_tools
        )
        res['extracted_json'] = extract_json(res['final_message'])

        with open('./function_calling_log.json', 'w', encoding = 'utf-8') as file:
            json.dump(res, file, ensure_ascii = False, indent = 4)