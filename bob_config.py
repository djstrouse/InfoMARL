from collections import namedtuple
from util.anneal import log_decay

experiment_name = 'bob_64_200k'

alice_experiment = 'alice_on_3x3'

# justification for experiment
'''
since delayed goal access experiments suggest MLP is capable, running for
more episodes: 200k instead of 100k
next exps: train even longer, widen layers, deepen, use shared layer for val/pol
'''

# RNNObserver agent variables
AgentParam = namedtuple('AgentParameters',
                       ['policy_layer_sizes',
                        'value_layer_sizes',
                        'learning_rate',
                        'use_RNN'])
agent_param = AgentParam(policy_layer_sizes = [64],
                         value_layer_sizes = [64],
                         learning_rate = 0.0001,
                         use_RNN = True)

# REINFORCE training variables
TrainingParam = namedtuple('TrainingParameters',
                          ['training_steps',
                           'entropy_scale',
                           'value_scale',
                           'discount_factor',
                           'max_episode_length',
                           'bob_goal_access'])
training_steps = 1000000 # 1M
training_param = TrainingParam(training_steps = training_steps,
                               entropy_scale = log_decay(.5, .01, num_episodes),
                               value_scale = .5,
                               discount_factor = .8,
                               max_episode_length = 100,
                               bob_goal_access = None)

def get_config():
    return agent_param, training_param, experiment_name, alice_experiment