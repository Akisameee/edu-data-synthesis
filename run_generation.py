from modules.planner import PlannerAgent
from modules.user_generate import GeneratorAgent

if __name__ == '__main__':

    generator = GeneratorAgent('gpt-4o')
    planner = PlannerAgent(
        'gpt-4o',
        generator
    )

    theme = {
        'scope': '纠错',
        'description': 'user（学生）对某个题目给出了错误的回答，assistant（智能体）找出错误并解释指正了user'
    }
    theme = {
        'scope': '答疑',
        'description': 'user（学生）对某个题目或者知识点存在疑问，assistant（智能体）帮助user解答疑问'
    }
    theme = {
        'scope': '判题',
        'description': 'user（教师或助教）给出一个题目和一个答案，assistant（智能体）帮助user判断答案正误并给出理由'
    }
    planner.run(
        scope = theme['scope'],
        task = theme['description']
    )