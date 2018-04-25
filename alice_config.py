from collections import namedtuple
from util.anneal import log_decay

experiment_name = 'alice_negative_competitive'

# justification for experiment
'''
testing state info optimization on vanilla 5x5 env
troubleshooting: state info strength / schedule, discount factor, training time
next: random transitions + interior goals = overshooting?
'''

# parameters to set up (fixed) computational graph
AgentParam = namedtuple('AgentParameters',
                       ['use_action_info',
                        'use_state_info'])
agent_param = AgentParam(use_action_info = False,
                         use_state_info = True)

# parameters fed as placeholders
TrainingParam = namedtuple('TrainingParameters',
                          ['training_steps',
                           'learning_rate',
                           'entropy_scale',
                           'value_scale',
                           'action_info_scale',
                           'state_info_scale',
                           'state_count_discount',
                           'discount_factor',
                           'max_episode_length'])
training_steps = 100000 # 100k
unregularized_steps = 10000 # 10k
training_param = TrainingParam(training_steps = training_steps,
                               learning_rate = .025,
                               entropy_scale = log_decay(.5, .005, training_steps),
                               value_scale = .5,
                               action_info_scale = None,
                               state_info_scale = [0]*unregularized_steps+[-.2]*int(training_steps-unregularized_steps),
                               state_count_discount = .999,
                               discount_factor = .85,
                               max_episode_length = 100)

def get_config():
    return agent_param, training_param, experiment_name