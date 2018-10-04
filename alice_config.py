from collections import namedtuple
from util.anneal import log_decay

experiment_name = 'alice_positive_state_cooperative'

# justification for experiment
'''
retraining alice with info added to return
'''

# parameters to set up (fixed) computational graph
AgentParam = namedtuple('AgentParameters',
                       ['use_action_info',
                        'use_state_info'])
agent_param = AgentParam(use_action_info = True,
                         use_state_info = False)

# parameters fed as placeholders
TrainingParam = namedtuple('TrainingParameters',
                          ['training_steps',
                           'learning_rate',
                           'entropy_scale',
                           'value_scale',
                           'action_info_scale',
                           'state_info_scale',
                           'state_count_discount',
                           'state_count_smoothing',
                           'discount_factor',
                           'max_episode_length'])
training_steps = 100000 # 100k
beta = .05
gamma = .9
training_param = TrainingParam(training_steps = training_steps,
                               learning_rate = .025,
                               entropy_scale = log_decay(.5, .005, training_steps),
                               value_scale = .5,
                               action_info_scale = beta,
                               state_info_scale = None,
                               state_count_discount = None,
                               state_count_smoothing = None,
                               discount_factor = gamma,
                               max_episode_length = 100)

def get_config():
    return agent_param, training_param, experiment_name