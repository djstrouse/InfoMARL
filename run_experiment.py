import tensorflow as tf
import numpy as np
import os
import sys
import pickle
import datetime
from shutil import copy
if "../" not in sys.path:
  sys.path.append("../") 
from envs.TwoGoalGridWorld import TwoGoalGridWorld
from agents.TabularREINFORCE import *
from training.REINFORCE import reinforce
from plotting.plot_episode_stats import *
from plotting.visualize_grid_world import *
import config

# initialize experiment using config.py
tf.reset_default_graph()
global_step = tf.Variable(0, name = "global_step", trainable = False)
env_param, agent_param, training_param, experiment_name = config.get_config()
env = TwoGoalGridWorld(env_param.shape,
                       env_param.r_correct,
                       env_param.r_incorrect,
                       env_param.r_step)
policy_estimator = PolicyEstimator(env, agent_param.policy_learning_rate)
value_estimator = ValueEstimator(env, agent_param.value_learning_rate)

# run experiment
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    stats = reinforce(env, policy_estimator, value_estimator,
                      num_episodes = training_param.num_episodes,
                      entropy_scale = training_param.entropy_scale,
                      beta = training_param.beta,
                      discount_factor = training_param.discount_factor)
    values = get_values(value_estimator, env, sess) # state X goal
    kls = get_kls(policy_estimator, env, sess) # state X goal
    action_probs = get_action_probs(policy_estimator, env, sess) # state X goal X action

# save experiment
Result = namedtuple('Result', ['episode_lengths', 'episode_rewards', 'values', 'kls', 'action_probs'])
result = Result(episode_lengths = stats.episode_lengths,
                episode_rewards = stats.episode_rewards,
                values = values, kls = kls, action_probs = action_probs)
results_directory = os.getcwd()+'/results/'
experiment_directory = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")+'_'+experiment_name+'/'
directory = results_directory + experiment_directory
if not os.path.exists(directory): os.makedirs(directory)
with open(directory+'results.pkl', 'wb') as output:
    pickle.dump(result, output, pickle.HIGHEST_PROTOCOL)

# copy config file to results directory to ensure experiment repeatable
copy(os.getcwd()+'/config.py', directory)
    
# plot experiment and save figures
FigureSizes = namedtuple('FigureSizes', ['figure', 'tick_label', 'axis_label', 'title'])
figure_sizes = FigureSizes(figure = (50,25),
                           tick_label = 40,
                           axis_label = 50,
                           title = 60)
plot_episode_stats(stats, figure_sizes, smoothing_window = 25, noshow = True, directory = directory)
k = 15
print('')
print('-'*k+'VALUES'+'-'*k)
plot_value_map(values, action_probs, env, figure_sizes, noshow = True, directory = directory)
print('')
print('-'*k+'KLS'+'-'*k)
plot_kl_map(kls, action_probs, env, figure_sizes, noshow = True, directory = directory)
print('')
print('-'*k+'POLICY'+'-'*k)
print_policy(action_probs, env)