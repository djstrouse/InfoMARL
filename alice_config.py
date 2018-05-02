from collections import namedtuple
from util.anneal import log_decay

experiment_name = 'alice_state_positive_cooperative'

# justification for experiment
'''
trying to get alice to overshoot, so testing various reg strengths
best info_reg strength: ~.025-.15 positive, ~.15-.35 negative 
troubleshooting: state info strength / schedule, training time, discount info into future?
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
training_steps = 500000 # 500k
unregularized_steps = 10000 # 10k
state_info_reg_strength = .15
training_param = TrainingParam(training_steps = training_steps,
                               learning_rate = .025,
                               entropy_scale = log_decay(.5, .005, training_steps),
                               value_scale = .5,
                               action_info_scale = None,
                               state_info_scale = [0]*unregularized_steps+[state_info_reg_strength]*int(training_steps-unregularized_steps),
                               state_count_discount = 1,
                               discount_factor = .8,
                               max_episode_length = 100)

def get_config():
    return agent_param, training_param, experiment_name