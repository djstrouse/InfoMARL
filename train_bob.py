import tensorflow as tf
import numpy as np
import os
import sys
import pickle
import datetime
import importlib
from collections import namedtuple
from shutil import copy
if "../" not in sys.path:
  sys.path.append("../") 
from envs.TwoGoalGridWorld import TwoGoalGridWorld
from agents.bob import RNNObserver
from training.REINFORCE_bob import reinforce
from plotting.plot_episode_stats import *
from plotting.visualize_grid_world import *
import env_config
import alice_config

Result = namedtuple('Result', ['episode_lengths', 'episode_rewards'])

def train_bob(config_extension = ''):
  
  config = importlib.import_module('bob_config'+config_extension)

  # initialize experiment using config.py
  tf.reset_default_graph()
  global_step = tf.Variable(0, name = "global_step", trainable = False)
  env_param, _ = env_config.get_config()
  agent_param, training_param, experiment_name = config.get_config()
  env = TwoGoalGridWorld(env_param.shape,
                         env_param.r_correct,
                         env_param.r_incorrect,
                         env_param.r_step,
                         env_param.goal_locs,
                         env_param.goal_dist)

  with tf.variable_scope('alice'):
    alice_agent_param, alice_training_param, alice_experiment_name = alice_config.get_config()
    alice = PolicyEstimator(env, alice_agent_param.policy_learning_rate)
    alice_saver = tf.train.Saver()

  with tf.variable_scope('bob'):
    bob = RNNObserver(env,
                      agent_param.policy_layer_sizes,
                      agent_param.value_layer_sizes,
                      agent_param.learning_rate)
    saver = tf.train.Saver()
  
  # run experiment
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    alice_saver.restore(sess, 'alice.ckpt')
    stats = reinforce(env, alice, bob,
                      num_episodes = training_param.num_episodes,
                      entropy_scale = training_param.entropy_scale,
                      value_scale = training_param.value_scale,
                      discount_factor = training_param.discount_factor,
                      max_episode_length = training_param.max_episode_length)
    # save session
    results_directory = os.getcwd()+'/results/'
    experiment_directory = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")+'_'+experiment_name+'/'
    directory = results_directory + experiment_directory
    save_path = saver.save(sess, directory+"bob.ckpt")
    print('')
    print("Model saved in path: %s" % save_path)
  
  # save experiment stats
  result = Result(episode_lengths = stats.episode_lengths,
                  episode_rewards = stats.episode_rewards)
  if not os.path.exists(directory): os.makedirs(directory)
  with open(directory+'results.pkl', 'wb') as output:
    pickle.dump(result, output, pickle.HIGHEST_PROTOCOL)
  
  # copy config file to results directory to ensure experiment repeatable
  copy(os.getcwd()+'/bob_config.py', directory)
  copy(os.getcwd()+'/alice_config.py', directory)
  copy(os.getcwd()+'/env_config.py', directory)
      
  # plot experiment and save figures
  FigureSizes = namedtuple('FigureSizes', ['figure', 'tick_label', 'axis_label', 'title'])
  figure_sizes = FigureSizes(figure = (50,25),
                             tick_label = 40,
                             axis_label = 50,
                             title = 60)
  plot_episode_stats(stats, figure_sizes, smoothing_window = 25, noshow = True, directory = directory)
  
  return

if __name__ == "__main__":
  train_bob()