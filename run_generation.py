import json

from modules.planner import Planner
from modules.utils import read_criterias

if __name__ == '__main__':

    criterias = read_criterias(
        './data/criteria/evaluation_metrics.json',
        './data/criteria/metrics_map.json'
    )

    models = ['qwen2.5-7b-instruct', 'qwen2.5-14b-instruct', 'qwen-max', 'deepseek-v3', 'deepseek-r1']
    planner = Planner(
        models
    )

    theme = {
        'task': '纠错',
        'description': 'user给出了某个题目和一个错误的回答，assistant找出错误并进行了解释和修改'
    }
    # theme = {
    #     'task': '答疑',
    #     'description': 'user（学生）对某个题目或者知识点存在疑问，assistant（智能体）帮助user解答疑问'
    # }
    # theme = {
    #     'task': '判题',
    #     'description': 'user（教师或助教）给出一个题目和一个答案，assistant（智能体）帮助user判断答案正误并给出理由'
    # }

    init_state = {
        'theme': theme,
        'criteria': criterias[theme['task']]
    }
    gen_res, trajectory = planner.run(init_state)
    print(gen_res)

    with open('./function_calling_log.json', 'w', encoding = 'utf-8') as file:
        json.dump(trajectory, file, ensure_ascii = False, indent = 4)