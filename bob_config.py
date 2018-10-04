from collections import namedtuple
from util.anneal import log_decay

experiment_name = 'bob_local_test'

alice_experiment = 'job17583555_task101_2018_05_17_1630_alice_negative_action_competitive_beta0.2_discount0.8_250k_KeyGame'

# justification for experiment
'''
running bob on alices trained with state info
'''

# parameters to set up (fixed) computational graph
AgentParam = namedtuple('AgentParameters',
                       ['shared_layer_sizes',
                        'policy_layer_sizes',
                        'value_layer_sizes',
                        'use_RNN'])
agent_param = AgentParam(shared_layer_sizes = [128],
                         policy_layer_sizes = [],
                         value_layer_sizes = [],
                         use_RNN = True)

# parameters fed as placeholders
TrainingParam = namedtuple('TrainingParameters',
                          ['training_steps',
                           'learning_rate',
                           'entropy_scale',
                           'value_scale',
                           'discount_factor',
                           'max_episode_length',
                           'bob_goal_access'])
training_steps = 200000 # 200k
training_param = TrainingParam(training_steps = training_steps,
                               learning_rate = 0.00005,
                               entropy_scale = log_decay(.05, .01, training_steps),
                               value_scale = .5,
                               discount_factor = .8,
                               max_episode_length = 100,
                               bob_goal_access = None)

def get_config():
    return agent_param, training_param, experiment_name, alice_experiment