import tensorflow as tf
import numpy as np
import os
import sys
import pickle
import datetime
import importlib
import glob
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

Result = namedtuple('Result', ['alice', 'bob'])
Stats = namedtuple('Stats', ['episode_lengths', 'episode_rewards', 'episode_kls'])

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
    bob = RNNObserver(env = env,
                      policy_layer_sizes = agent_param.policy_layer_sizes,
                      value_layer_sizes = agent_param.value_layer_sizes,
                      learning_rate = agent_param.learning_rate,
                      use_RNN = agent_param.use_RNN)
    saver = tf.train.Saver()
  
  # run experiment
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    alice_saver.restore(sess, 'alice.ckpt')
    alice_stats, bob_stats = reinforce(env, alice, bob,
                                       num_episodes = training_param.num_episodes,
                                       entropy_scale = training_param.entropy_scale,
                                       value_scale = training_param.value_scale,
                                       discount_factor = training_param.discount_factor,
                                       max_episode_length = training_param.max_episode_length,
                                       bob_goal_access = training_param.bob_goal_access)
    # save session
    results_directory = os.getcwd()+'/results/'
    experiment_directory = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")+'_'+experiment_name+'/'
    directory = results_directory + experiment_directory
    if not os.path.exists(directory+'bob/'): os.makedirs(directory+'bob/')
    save_path = saver.save(sess, directory+'bob/bob.ckpt')
  
  # save experiment stats
  if not os.path.exists(directory): os.makedirs(directory)
  with open(directory+'results.pkl', 'wb') as output:
    # copy to locally-defined Stats objects to make pickle happy
    a = Stats(episode_lengths = alice_stats.episode_lengths,
              episode_rewards = alice_stats.episode_rewards,
              episode_kls = alice_stats.episode_kls)
    b = Stats(episode_lengths = bob_stats.episode_lengths,
              episode_rewards = bob_stats.episode_rewards,
              episode_kls = bob_stats.episode_kls)
    pickle.dump(Result(alice = a, bob = b), output, pickle.HIGHEST_PROTOCOL)
  
  # copy config file to results directory to ensure experiment repeatable
  copy(os.getcwd()+'/bob_config.py', directory)
  copy(os.getcwd()+'/alice_config.py', directory)
  copy(os.getcwd()+'/env_config.py', directory)
  
  # copy alice checkpoint used
  if not os.path.exists(directory+'alice/'): os.makedirs(directory+'alice/')
  for file in glob.glob(os.getcwd()+'/alice.ckpt*'): copy(file, directory+'alice/')
  copy(os.getcwd()+'/checkpoint', directory+'alice/')
      
  # plot experiment and save figures
  FigureSizes = namedtuple('FigureSizes', ['figure', 'tick_label', 'axis_label', 'title'])
  figure_sizes = FigureSizes(figure = (50,25),
                             tick_label = 40,
                             axis_label = 50,
                             title = 60)
  plot_episode_stats(bob_stats, figure_sizes, smoothing_window = 25, noshow = True, directory = directory)
  print('\n\nResults saved in {}'.format(directory))
  
  return

if __name__ == "__main__":
  train_bob()