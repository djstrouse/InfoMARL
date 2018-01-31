from collections import namedtuple
from util.anneal import log_decay

experiment_name = 'test'

# justification for experiment
'''
just seeing if bob works
'''

# RNNObserver agent variables
AgentParam = namedtuple('AgentParameters',
                       ['policy_layer_sizes',
                        'value_layer_sizes',
                        'learning_rate'])
agent_param = AgentParam(policy_layer_sizes = [128],
                         value_layer_sizes = [128],
                         learning_rate = 0.025)

# REINFORCE training variables
TrainingParam = namedtuple('TrainingParameters',
                          ['num_episodes',
                           'entropy_scale',
                           'value_scale',
                           'discount_factor',
                           'max_episode_length'])
num_episodes = 1000
training_param = TrainingParam(num_episodes = num_episodes,
                               entropy_scale = log_decay(.5, .005, num_episodes),
                               value_scale = .5,
                               discount_factor = .9,
                               max_episode_length = 100)

def get_config():
    return agent_param, training_param, experiment_name