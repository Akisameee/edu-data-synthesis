# import json
# from tqdm import tqdm
# import asyncio

# from modules.models import get_model
# from modules.workflow import *
# from modules.optimizer import *
# from modules.datas import *

# w1 = EvaluationWorkflow()
# w1.add_node('evaluate_0', Evaluate(get_model('deepseek-v3')))
# w1.add_node('evaluate_1', Evaluate(get_model('deepseek-r1')))
# w1.add_node('evaluate_2', Evaluate(get_model('gpt-4o')))
# w1.add_node('evaluate_3', Evaluate(get_model('qwen-max')))
# w1.add_node('evaluate_4', Evaluate(get_model('deepseek-v3')))
# w1.add_node('evaluate_5', Evaluate(get_model('deepseek-v3')))
# w1.add_node('aggregate_0', EvaluationAggregation(get_model('deepseek-v3')))

# w1.add_edge('input', 'evaluate_0')
# w1.add_edge('input', 'evaluate_1')
# w1.add_edge('input', 'evaluate_2')
# w1.add_edge('input', 'evaluate_3')

# w1.add_edge('evaluate_0', 'aggregate_0')
# w1.add_edge('evaluate_1', 'aggregate_0')
# w1.add_edge('evaluate_2', 'aggregate_0')
# w1.add_edge('evaluate_3', 'aggregate_0')
# w1.add_edge('evaluate_4', 'aggregate_0')

# w1.add_edge('aggregate_0', 'output')
# print(w1.get_topo_order())


# w2 = EvaluationWorkflow()
# w2.add_node('eval_0', Evaluate(get_model('deepseek-v3')))
# w2.add_node('eval_1', Evaluate(get_model('deepseek-r1')))
# w2.add_node('eval_2', Evaluate(get_model('gpt-4o')))
# w2.add_node('eval_3', Evaluate(get_model('qwen-max')))
# w2.add_node('eval_4', Evaluate(get_model('deepseek-v3')))
# w2.add_node('eval_5', Evaluate(get_model('deepseek-v3')))
# w2.add_node('aggregate_0', EvaluationAggregation(get_model('deepseek-v3')))

# w2.add_edge('input', 'eval_0')
# w2.add_edge('input', 'eval_1')
# w2.add_edge('input', 'eval_2')
# w2.add_edge('input', 'eval_3')

# w2.add_edge('eval_0', 'aggregate_0')
# w2.add_edge('eval_1', 'aggregate_0')
# w2.add_edge('eval_2', 'aggregate_0')
# w2.add_edge('eval_3', 'aggregate_0')
# w2.add_edge('eval_4', 'aggregate_0')

# w2.add_edge('aggregate_0', 'output')

# # altered
# w2.add_node('aggregate_1', EvaluationAggregation(get_model('deepseek-r1')))
# w2.add_edge('eval_4', 'aggregate_1')
# w2.add_edge('eval_5', 'aggregate_1')
# w2.add_edge('aggregate_1', 'aggregate_0')
# print(w2.get_topo_order())

# print(w1 == w2)

# print(w1.to_tuple())
# print(w2.to_tuple())
# print(w1.equal(w2))

# print(w1.sub_nec.to_tuple())
# print(w2.sub_nec.to_tuple())
# print(w1.sub_nec.equal(w2.sub_nec))

from jinja2 import Environment
env = Environment()
template_content = "Hello {{ name }} {test}"
parsed_content = env.parse(template_content)
print(parsed_content)