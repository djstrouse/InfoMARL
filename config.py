from collections import namedtuple
from util.anneal import log_decay

experiment_name = 'test'

# TwoGoalGridWorld environment variables
EnvParam = namedtuple('EnvironmentParameters',
                     ['shape',
                      'r_correct',
                      'r_incorrect',
                      'r_step'])
env_param = EnvParam(shape = [3,3],
                     r_correct = +1,
                     r_incorrect = -1,
                     r_step = 0.)

# TabularREINFORCE agent variables
AgentParam = namedtuple('AgentParameters',
                       ['policy_learning_rate',
                        'value_learning_rate'])
agent_param = AgentParam(policy_learning_rate = 0.025,
                         value_learning_rate = 0.05)

# REINFORCE training variables
TrainingParam = namedtuple('TrainingParameters',
                          ['num_episodes',
                           'entropy_scale',
                           'beta',
                           'discount_factor'])
num_episodes = 100
training_param = TrainingParam(num_episodes = num_episodes,
                               entropy_scale = log_decay(.2, .01, num_episodes),
                               beta = 0,
                               discount_factor = .8)

def get_config():
    return env_param, agent_param, training_param, experiment_name