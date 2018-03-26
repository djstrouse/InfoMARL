import tensorflow as tf
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
from agents.alice import TabularREINFORCE, get_values, get_kls, get_action_probs
from training.REINFORCE_alice import reinforce
from plotting.plot_episode_stats import plot_episode_stats
from plotting.visualize_grid_world import plot_value_map, plot_kl_map, print_policy

Result = namedtuple('Result',
                   ['episode_lengths', 'episode_rewards', 'values', 'kls', 'action_probs'])

def train_alice(alice_config_ext = '', env_config_ext = '',
                exp_name_ext = '', exp_name_prefix = '', results_directory = None):
  
  if results_directory is None: results_directory = os.getcwd()+'/results/'
  
  config = importlib.import_module('alice_config'+alice_config_ext)
  env_config = importlib.import_module('env_config'+env_config_ext)

  # initialize experiment using config.py
  tf.reset_default_graph()
  global_step = tf.Variable(0, name = "global_step", trainable = False)
  env_param, env_exp_name_ext = env_config.get_config()
  training_param, experiment_name = config.get_config()
  experiment_name = experiment_name + env_exp_name_ext + exp_name_ext
  env = TwoGoalGridWorld(shape = env_param.shape,
                         r_correct = env_param.r_correct,
                         r_incorrect = env_param.r_incorrect,
                         r_step = env_param.r_step,
                         r_wall = env_param.r_wall,
                         p_rand = env_param.p_rand,
                         goal_locs = env_param.goal_locs,
                         goal_dist = env_param.goal_dist)
  with tf.variable_scope('alice'):
    alice = TabularREINFORCE(env)
  saver = tf.train.Saver()
  
  # run experiment
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    stats = reinforce(env = env,
                      agent = alice,
                      training_steps = training_param.training_steps,
                      learning_rate = training_param.learning_rate,
                      entropy_scale = training_param.entropy_scale,
                      value_scale = training_param.value_scale,
                      info_scale = training_param.info_scale,
                      discount_factor = training_param.discount_factor,
                      max_episode_length = training_param.max_episode_length)
    values = get_values(alice, env, sess) # state X goal
    kls = get_kls(alice, env, sess) # state X goal
    action_probs = get_action_probs(alice, env, sess) # state X goal X action
    # save session
    experiment_directory = exp_name_prefix+datetime.datetime.now().strftime("%Y_%m_%d_%H%M")+'_'+experiment_name+'/'
    directory = results_directory + experiment_directory
    save_path = saver.save(sess, directory+"alice.ckpt")
    print('')
    print("Model saved in path: %s" % save_path)
  
  # save experiment stats
  result = Result(episode_lengths = stats.episode_lengths,
                  episode_rewards = stats.episode_rewards,
                  values = values, kls = kls, action_probs = action_probs)
  if not os.path.exists(directory): os.makedirs(directory)
  with open(directory+'results.pkl', 'wb') as output:
    pickle.dump(result, output, pickle.HIGHEST_PROTOCOL)
  
  # copy config file to results directory to ensure experiment repeatable
  copy(os.getcwd()+'/alice_config'+alice_config_ext+'.py', directory+'alice_config.py')
  copy(os.getcwd()+'/env_config'+env_config_ext+'.py', directory+'env_config.py')
      
  # plot experiment and save figures
  FigureSizes = namedtuple('FigureSizes', ['figure', 'tick_label', 'axis_label', 'title'])
  figure_sizes = FigureSizes(figure = (50,25),
                             tick_label = 40,
                             axis_label = 50,
                             title = 60)
  plot_episode_stats(stats, figure_sizes, noshow = True, directory = directory)
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
  print('')
  
  return

if __name__ == "__main__":
  train_alice()