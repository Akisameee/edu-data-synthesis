Provided Inputs:

- `scenario`:
  - task: Question Generation
  - description: The user provides information such as educational level, subject, and knowledge points, and the assistant generates questions according to the given requirements.
- `messages`:
[{'role': 'user', 'content': '学科:基础医学\n学制级别:硕士\n知识点：肌肉收缩的生理机制\n问题类型：多选题\n.请根据学科,学制级别,知识点,问题类型生成一个对应的问题。"问题"和"提供的思路":,以json格式返回'}, {'role': 'assistant', 'content': '```json\n{\n  "问题": "关于肌肉收缩的生理机制，以下哪些描述是正确的？",\n  "提供的思路": [\n    "肌肉收缩的基本单位是肌原纤维，主要由肌动蛋白和肌球蛋白组成。",\n    "钙离子在肌肉收缩过程中起关键作用，它通过与肌钙蛋白结合，暴露肌动蛋白上的结合位点。",\n    "ATP是肌肉收缩的直接能源，它通过水解为ADP和磷酸释放能量。",\n    "神经冲动的传递通过神经-肌肉接头，导致肌质网释放钙离子。",\n    "肌肉收缩时，肌球蛋白头部与肌动蛋白结合，形成横桥，产生收缩力。"\n  ]\n}\n```'}]
- `criteria`:
  - [0]
    - name: Basic Factual Accuracy
    - description: Are objective facts such as concept definitions, formulas, dates, terminology, code syntax, legal clauses correctly presented?
    - rules:
      - 9-10: All stated factual elements (definitions, formulas, dates, terms, syntax, etc.) are completely accurate.
      - 7-8: Vast majority of facts are correct; possibly contains very minor, non-critical typos or omissions.
      - 5-6: Most facts are correct, but there are some notable factual errors that require review.
      - 3-4: Contains several or key factual inaccuracies; information is not trustworthy.
      - 1-2: Riddled with factual errors; information is completely incorrect or misleading.