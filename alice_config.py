from collections import namedtuple
from util.anneal import log_decay

experiment_name = 'unregularized_ambivalent'

# justification for experiment
'''
training alice on 4x4
'''

# TabularREINFORCE agent variables
AgentParam = namedtuple('AgentParameters',
                       ['policy_learning_rate',
                        'value_learning_rate'])
agent_param = AgentParam(policy_learning_rate = 0.025,
                         value_learning_rate = 0.0125)

# REINFORCE training variables
TrainingParam = namedtuple('TrainingParameters',
                          ['training_steps',
                           'entropy_scale',
                           'beta',
                           'discount_factor',
                           'max_episode_length'])
training_steps = 400000
training_param = TrainingParam(training_steps = training_steps,
                               entropy_scale = log_decay(.5, .005, num_episodes),
                               beta = 0,
                               discount_factor = .85,
                               max_episode_length = 100)

def get_config():
    return agent_param, training_param, experiment_name