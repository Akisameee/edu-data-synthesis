user_generate_template = \
'''
你是一个教育领域大模型的用户，请使用元数据，在历史数据的基础上模拟用户在给定的教育场景下向大模型助手发出一次请求/询问对话。
可使用<meta_data>符号在content中对元数据进行引用。
以json的单次对话格式返回，如下所示：
```json{{"role": "user","content": "<请求/询问内容>"}}```

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
我将向你提供一段教育领域下特定场景的对话，请根据所给定的所有评估指标及其评分细则对所给的回答进行评分并给出理由。
以json的格式返回，例如：
```json[{{"criterion": "<评估指标1名称>", "score": <得分>, "reason": "<理由>"}}, {{"criterion": "<评估指标2名称>", ...}}, ...]```

场景：
{scenario}
对话：
{message}
评估指标: 
{criteria}
'''

review_template = \
'''
我将向你提供一段教育领域下特定场景的对话，请根据所给定的所有评估指标及其评分细则对这段对话的assistant提出改进意见。
以json的列表格式返回，例如：
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
以json的索引对话格式返回（只返回修改的对话即可），例如：
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

planning_template = \
'''
你是一个规划智能体，你需要通过调用工具执行教育领域{task}场景下的数据生成过程。

考虑一个马尔可夫决策过程，你需要根据现有的状态（生成的结果、中间过程等）决策并执行一次动作（调用工具）。
在每次执行动作后，你会得到：
    - state：状态信息
        - scenario：本条数据对应的特定教育场景
        - criteria：针对该教育场景的评估指标
        - meta_data：生成本条数据使用的元数据
        - message：生成的对话数据
        - critique：针对message的改进意见
    - available_actions：当前状态下可用的动作（工具）
    - available_models：可用的模型名称

大部分动作需要依靠大语言模型的生成能力，调用大模型也需要产生一定的开销。
目标（优先级从高到低）：
    1. 生成message对话数据
        - 调用user_generate和assistant_generate生成对话数据
            - 不同模型的生成性能不同，原始数据的生成很大程度上决定了数据质量的上限
    2. 尽可能将对话数据在所有模型的所有评估指标上的表现提升至满分10分
        - 通过evaluate评估目前message得分，决定是否继续改进
            - 不同模型的评估性能不同，各指标以所有模型的最低分为准
        - 通过review和refine对现有message进行改进
            - 不同模型的反馈/改进性能不同
        - 生成多轮对话扩充message内容
    3. 尽可能降低开销

初始状态：
{init_state}
'''