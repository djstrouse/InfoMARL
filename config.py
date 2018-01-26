from collections import namedtuple
from util.anneal import log_decay

experiment_name = 'small_negative'

# justification for experiment
'''
lowering magnitude of beta since policies suboptimal
'''

# TwoGoalGridWorld environment variables
EnvParam = namedtuple('EnvironmentParameters',
                     ['shape',
                      'r_correct',
                      'r_incorrect',
                      'r_step'])
env_param = EnvParam(shape = [5,5],
                     r_correct = +1,
                     r_incorrect = -1,
                     r_step = 0.)

# TabularREINFORCE agent variables
AgentParam = namedtuple('AgentParameters',
                       ['policy_learning_rate',
                        'value_learning_rate'])
agent_param = AgentParam(policy_learning_rate = 0.025,
                         value_learning_rate = 0.025)

# REINFORCE training variables
TrainingParam = namedtuple('TrainingParameters',
                          ['num_episodes',
                           'entropy_scale',
                           'beta',
                           'discount_factor',
                           'max_episode_length'])
num_episodes = 100000
training_param = TrainingParam(num_episodes = num_episodes,
                               entropy_scale = log_decay(.5, .005, num_episodes),
                               beta = -.025,
                               discount_factor = .9,
                               max_episode_length = 100)

def get_config():
    return env_param, agent_param, training_param, experiment_name