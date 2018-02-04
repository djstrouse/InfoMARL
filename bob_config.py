from collections import namedtuple
from util.anneal import log_decay

experiment_name = 'bob_with_cooperative_alice_5x5'

alice_experiment = '2018_02_03_0950_small_positive_5x5'

# justification for experiment
'''
retraining bob with cooperative alice for more episodes to see if performance gap widens
could try next: widening NN layers, even longer training, altering spawn locations
'''

# RNNObserver agent variables
AgentParam = namedtuple('AgentParameters',
                       ['policy_layer_sizes',
                        'value_layer_sizes',
                        'learning_rate',
                        'use_RNN'])
agent_param = AgentParam(policy_layer_sizes = [64, 64],
                         value_layer_sizes = [64, 64],
                         learning_rate = 0.0001,
                         use_RNN = True)

# REINFORCE training variables
TrainingParam = namedtuple('TrainingParameters',
                          ['num_episodes',
                           'entropy_scale',
                           'value_scale',
                           'discount_factor',
                           'max_episode_length',
                           'bob_goal_access'])
num_episodes = 100000
training_param = TrainingParam(num_episodes = num_episodes,
                               entropy_scale = log_decay(.5, .005, num_episodes),
                               value_scale = .5,
                               discount_factor = .9,
                               max_episode_length = 100,
                               bob_goal_access = None)

def get_config():
    return agent_param, training_param, experiment_name, alice_experiment