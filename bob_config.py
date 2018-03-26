from collections import namedtuple
from util.anneal import log_decay

experiment_name = 'bob_shared128_250k'

alice_experiment = '2018_03_26_1407_alice_positive_cooperative_5x5'

# justification for experiment
'''
testing code refactoring on 5x5
next exps: train longer, widen layers, deepen, REINFORCE -> actor-critic,
           switch between freeze RNN + train NN and train RNN + freeze NN,
           turn up value scale, epsilon exploration
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
                               discount_factor = .9,
                               max_episode_length = 100,
                               bob_goal_access = None)

def get_config():
    return agent_param, training_param, experiment_name, alice_experiment