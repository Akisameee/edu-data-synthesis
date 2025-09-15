Provided Inputs:

- `scenario`:
  - task: Personalized Learning Support
  - description: The user provides a student profile, and the assistant designs a learning path.
- `messages`:
[{'role': 'user', 'content': "学生画像：学生的画像：{'姓名': '小明', '年级': '小学四年级', '兴趣': ['观察自然', '阅读科普书籍'], '学习风格': '动手实践型', '能力水平': '理解力较强，但记忆力和专注度需要加强', '性格特点': '好奇心强，喜欢提出问题'}\n请根据学生画像，为每位学生生成 ‘个性化的学习内容或任务’:,以json格式返回"}, {'role': 'assistant', 'content': '根据提供的学生画像，为小明设计的个性化学习内容或任务如下所示。这些建议旨在利用他的兴趣（观察自然、阅读科普书籍）和学习风格（动手实践型），同时考虑到需要加强的能力（记忆力和专注度），以及性格特点（好奇心强，喜欢提问）。\n\n```json\n{\n  "姓名": "小明",\n  "年级": "小学四年级",\n  "个性化学习内容或任务": [\n    {\n      "类型": "实验活动",\n      "描述": "参与一次户外植物探索之旅，记录不同种类植物的特点，并尝试种植一种植物。",\n      "目的": "通过实践活动加深对自然科学的兴趣，同时锻炼观察力和记忆力。",\n      "建议资源": ["《小小植物学家手册》", "种子包"]\n    },\n    {\n      "类型": "互动式阅读",\n      "描述": "选择一本关于自然现象的科普书，每天固定时间阅读一章，并在读完后向家长或老师提出至少一个问题。",\n      "目的": "培养持续阅读习惯，提高信息理解能力及提问技巧。",\n      "建议书籍": ["《奇妙的天气》", "《地球的故事》"]\n    },\n    {\n      "类型": "记忆游戏",\n      "描述": "使用卡片形式制作一套关于动植物知识的记忆卡牌游戏，在家庭成员之间进行比赛。",\n      "目的": "以游戏方式增强记忆能力，增加与家人之间的互动。",\n      "材料准备": ["空白卡片", "彩色笔"]\n    },\n    {\n      "类型": "项目研究",\n      "描述": "选定一个感兴趣的自然主题（如昆虫、岩石等），搜集资料并制作成PPT或者海报形式展示给班级同学。",\n      "目的": "促进深入学习某一领域知识的同时，锻炼表达能力和团队协作精神。",\n      "辅助工具": ["互联网搜索", "演示文稿软件"]\n    }\n  ]\n}\n```\n\n这样的安排不仅能够激发小明的学习兴趣，还能帮助他在享受学习的过程中逐步提升自己的各项技能。'}]
- `criterion`:
  - name: Instruction Following & Task Completion
  - description: Did it fully understand and execute the user's instruction? Was the core task (e.g., solving problems, error correction, question generation) completed? Is the output formatting correct?
  - rules:
    - 9-10: Fully understood and precisely executed all instructions; achieved core task with perfect accuracy; output format is fully compliant.
    - 7-8: Accurately understood main instructions and correctly completed the task; core goals are well achieved; format is mostly correct with only minor omissions or deviations.
    - 5-6: Understood the general intent but may miss some details; task largely completed but with some inaccuracies or omissions; formatting attempts present but with notable flaws.
    - 3-4: Misunderstood part of the instruction; low task completion or major errors; formatting mostly incorrect.
    - 1-2: Completely misunderstood or ignored instructions; task not completed or totally incorrect; formatting is chaotic or irrelevant.