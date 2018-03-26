from collections import namedtuple
from util.anneal import log_decay

experiment_name = 'alice_negative_competitive'

# justification for experiment
'''
testing code refactoring on 5x5
'''

# parameters to set up (fixed) computational graph
# note: tabular Alice has no parameters of this type

# parameters fed as placeholders
TrainingParam = namedtuple('TrainingParameters',
                          ['training_steps',
                           'learning_rate',
                           'entropy_scale',
                           'value_scale',
                           'info_scale',
                           'discount_factor',
                           'max_episode_length'])
training_steps = 100000 # 100k
training_param = TrainingParam(training_steps = training_steps,
                               learning_rate = .025,
                               entropy_scale = log_decay(.5, .005, training_steps),
                               value_scale = .5,
                               info_scale = -.025,
                               discount_factor = .9,
                               max_episode_length = 100)

def get_config():
    return training_param, experiment_name