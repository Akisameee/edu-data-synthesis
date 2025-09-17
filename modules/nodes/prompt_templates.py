user_generate_template = \
'''
你是一个教育领域大模型的用户，请围绕元数据，在历史数据的基础上模拟用户在给定的教育场景下向大模型助手发出一次请求/询问对话。
以json的单次对话格式返回，如下所示：
```json{{"role": "user","content": "<请求/询问内容>"}}```
若对话历史未提到元数据或为空，则需对元数据进行引用。可使用[meta_data]符号在请求/询问内容中对元数据进行引用，例如"[meta_data]以上数据...，请你..."。
进行引用后无需复述元数据中的内容

场景:
{scenario}
元数据:
{meta_data}
对话历史：
{message}
'''

system_template = \
'''
你是一个教育领域的智能助手，帮助用户完成{task}任务，你的回复需要满足以下评估指标：
{criteria}
'''

evaluation_template = \
'''
你是一名数据评分员，我将向你提供一段教育领域下特定场景的对话，请你根据所给定的所有评估指标及其评分细则对所给的回答进行评分并给出原因。
请按照每个评估指标中的评分细则严格地进行评分，给出的原因需要结合对话原文具体说明满足了评分指标的哪一条。
最终结果以JSON的格式返回，例如：
```json[{{"criterion": "<评估指标1名称>", "score": <得分>, "reason": <原因>}}, {{"criterion": "<评估指标2名称>", "score": <得分>, "reason": <原因>}}, ...]```

场景：
{scenario}
对话：
{message}
评估指标: 
{criteria}
'''

evaluation_cl_template = \
'''
你是一名数据评分员，我将向你提供一段教育领域下特定场景的对话，请你根据所给定的所有评估指标及其评分细则对所给的回答进行评分并给出原因。
请参考评估样例的打分，按照每个评估指标中的评分细则严格地进行评分，给出的原因需要结合对话原文具体说明满足了评分指标的哪一条。
以JSON的格式返回，例如：
```json[{{"criterion": "<评估指标1名称>", "score": <得分>, "reason": <原因>}}, {{"criterion": "<评估指标2名称>", "score": <得分>, "reason": <原因>}}, ...]```

场景：
{scenario}

评估指标: 
{criteria}

评估样例：
{samples}

对话：
{message}
'''

evaluation_single_template = \
'''
你是一名数据评分员，我将向你提供一段教育领域下特定场景的对话，请你根据所给定的评估指标及其评分细则对所给的回答进行评分并给出原因。
请按照评估指标中的评分细则严格地进行评分，给出的原因需要结合对话原文具体说明满足了评分指标的哪一条。
以JSON的格式返回，例如：
```json{{"criterion": "<评估指标1名称>", "score": <得分>, "reason": <原因>}}```

场景：
{scenario}
对话：
{message}
评估指标: 
{criterion}
'''

review_template = \
'''
我将向你提供一段教育领域下特定场景的对话，请根据所给定的所有评估指标及其评分细则对这段对话的assistant提出改进意见。
以JSON的列表格式返回，例如：
```json["<改进意见1>", "<改进意见2>", ...]```

场景：
{scenario}
对话：
{message}
评估指标: 
{criteria}
'''

refine_template = \
'''
我将向你提供一段教育领域下特定场景的对话以及针对其的改进意见，请根据改进意见对原对话中assistant的回应进行改进。
以JSON的索引对话格式返回（只返回修改的对话即可），例如：
```json{{"<对话索引>": {{"role": "assistant", "content": "<改进后的回复内容>"}}, "<对话索引>": {{"role": "assistant", "content": "<改进后的回复内容>"}}, ...}}```

场景：
{scenario}
对话：
{message}
可改进的对话索引：
{assistant_idxs}
改进意见：
{critique}
'''

response_aggregate_template = \
'''
我将向你提供一组在教育领域下特定场景的对话，请汇总所有回复内容整合成一个新的回复。
不要对给出的对话内容简单地合并，而应该总结、提炼并优化出一个更完美的回复。
以JSON的单次对话格式返回，例如：
```json{{"role": "assistant", "content": "<汇总的回复内容>"}}```

场景：
{scenario}
对话历史：
{history}
'''

evaluation_aggregate_template = \
'''
我将向你提供一个在教育领域下特定场景的对话以及一组大模型对于这段对话的评分，请汇总每个评估指标的得分和理由整合一个最终评价。
注意观察评分与原因是否和评估指标中的评分标准对应、是与否与对话内容相对应。
以JSON的格式返回，例如：
```json[{{"criterion": "<评估指标1名称>", "score": <得分>, "reason": <原因>}}, {{"criterion": "<评估指标2名称>", "score": <得分>, "reason": <原因>}}, ...]```

场景：
{scenario}
对话：
{message}
评估指标：
{criterias}
'''

evaluation_voting_template = \
'''
我将向你提供一个在教育领域下特定场景的对话以及一组大模型对于这段对话的评分，请选择一个你认为最合理的评分。
你需要观察并分析评分与原因是否和评估指标中的评分标准对应、是与否与对话内容相对应，最终给出你的选项。
结果以\\boxed{{<你的选项>}}给出（大写字母）

场景：
{scenario}
对话：
{message}
评估指标：
{criteria}
评分：
'''

debate_template = \
'''
Contexts: 
{contexts}
Your response: 
{self_response}
Reponses from other agents: 
{other_responses}
Use opinions from other agents carefully as additional advice, provide an updated answer with same format.
'''