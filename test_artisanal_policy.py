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
from agents.alice import TabularREINFORCE, get_kls, get_action_probs
from training.REINFORCE_alice import reinforce
from plotting.plot_episode_stats import plot_episode_stats
from plotting.visualize_grid_world import plot_kl_map, plot_lso_map, plot_state_densities, print_policy

Result = namedtuple('Result',
                   ['episode_lengths', 'episode_rewards', 'values',
                    'action_kls', 'log_state_odds',
                    'action_probs', 'state_goal_counts'])

def train_alice(alice_config_ext = '', env_config_ext = '',
                exp_name_ext = '', exp_name_prefix = '', results_directory = None):
  
  if results_directory is None: results_directory = os.getcwd()+'/results/'
  
  config = importlib.import_module('alice_config'+alice_config_ext)
  env_config = importlib.import_module('env_config'+env_config_ext)
  
  # run training, and if nans, creep in, train again until they don't
  success = False
  
  while not success:
    # initialize experiment using config.py
    tf.reset_default_graph()
    #global_step = tf.Variable(0, name = "global_step", trainable = False)
    env_param, env_exp_name_ext = env_config.get_config()
    agent_param, training_param, experiment_name = config.get_config()
    experiment_name = experiment_name + env_exp_name_ext + exp_name_ext
    env = TwoGoalGridWorld(shape = env_param.shape,
                           r_correct = env_param.r_correct,
                           r_incorrect = env_param.r_incorrect,
                           r_step = env_param.r_step,
                           r_wall = env_param.r_wall,
                           p_rand = env_param.p_rand,
                           goal_locs = env_param.goal_locs,
                           goal_dist = env_param.goal_dist)
    print('Initialized environment.')
    with tf.variable_scope('alice'):
      policy = np.zeros((env.nG, env.nS, env.nA))
      for g in range(env.nG):
        for s in range(env.nS):
          if g == 0: # left goal
            if (s % env.max_x) == 0: # under it
              policy[g,s,env.action_to_index['UP']] = 1
            else:
              policy[g,s,env.action_to_index['LEFT']] = 1
          elif g == 1: # right goal
            if (s % env.max_x) == (env.max_x - 1): # under it
              policy[g,s,env.action_to_index['UP']] = 1
            else:
              policy[g,s,env.action_to_index['RIGHT']] = 1
      alice = TabularREINFORCE(env,
                               use_action_info = agent_param.use_action_info,
                               use_state_info = agent_param.use_state_info,
                               policy = policy)
      print('Initialized agent.')
    #saver = tf.train.Saver()
    
    # run experiment
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      stats, success = reinforce(env = env,
                                 agent = alice,
                                 training_steps = training_param.training_steps,
                                 learning_rate = training_param.learning_rate,
                                 entropy_scale = training_param.entropy_scale,
                                 value_scale = training_param.value_scale,
                                 action_info_scale = training_param.action_info_scale,
                                 state_info_scale = training_param.state_info_scale,
                                 state_count_discount = training_param.state_count_discount,
                                 discount_factor = training_param.discount_factor,
                                 max_episode_length = training_param.max_episode_length)
      if success: 
        print('Finished training.')
        #values = get_values(alice, env, sess) # state X goal
        values = None
        print('Extracted values.')
        if alice.use_action_info:
          action_kls = get_kls(alice, env, sess) # state X goal
          print('Extracted kls.')
        else:
          action_kls = None
        if alice.use_state_info:
          ps_g = stats.state_goal_counts / np.sum(stats.state_goal_counts, axis = 0)
          ps = np.sum(stats.state_goal_counts, axis = 1) / np.sum(stats.state_goal_counts)
          ps = np.expand_dims(ps, axis = 1)
          lso = np.log2(ps_g/ps) # state X goal
          print('Extracted log state odds.')
    #      lso1 = get_log_state_odds(alice, env, stats.state_goal_counts, sess)
        else:
          lso = None
          
        action_probs = get_action_probs(alice, env, sess) # state X goal X action
        print('Extracted policy.')
        # save session
        experiment_directory = exp_name_prefix+datetime.datetime.now().strftime("%Y_%m_%d_%H%M")+'_'+experiment_name+'/'
        directory = results_directory + experiment_directory
        #save_path = saver.save(sess, directory+"alice.ckpt")
        #print('')
        #print("Model saved in path: %s" % save_path)
      else:
        print('Unsucessful run - restarting.')
        f = open('error.txt','a')
        d = datetime.datetime.now().strftime("%A, %B %d, %I:%M:%S %p")
        f.write("{}: experiment '{}' failed and reran\n".format(d, exp_name_prefix+experiment_name))
        f.close()
  
  # save experiment stats
  result = Result(episode_lengths = stats.episode_lengths,
                  episode_rewards = stats.episode_rewards,
                  values = values,
                  action_kls = action_kls,
                  log_state_odds = lso,
                  action_probs = action_probs,
                  state_goal_counts = stats.state_goal_counts)
  if not os.path.exists(directory): os.makedirs(directory)
  with open(directory+'results.pkl', 'wb') as output:
    pickle.dump(result, output, pickle.HIGHEST_PROTOCOL)
  print('Saved stats.')
  
  # copy config file to results directory to ensure experiment repeatable
  copy(os.getcwd()+'/alice_config'+alice_config_ext+'.py', directory+'alice_config.py')
  copy(os.getcwd()+'/env_config'+env_config_ext+'.py', directory+'env_config.py')
  print('Copied configs.')
      
  # plot experiment and save figures
  FigureSizes = namedtuple('FigureSizes', ['figure', 'tick_label', 'axis_label', 'title'])
  figure_sizes = FigureSizes(figure = (50,25),
                             tick_label = 40,
                             axis_label = 50,
                             title = 60)
  
  steps_per_reward, _, action_info, state_info  = plot_episode_stats(stats,
                                                                     figure_sizes,
                                                                     noshow = True,
                                                                     directory = directory)
  k = 15
  print('')
  print('-'*k+'VALUES'+'-'*k)
  #plot_value_map(values, action_probs, env, figure_sizes, noshow = True, directory = directory)
  if action_kls is not None:
    print('')
    print('-'*k+'KLS'+'-'*k)
    plot_kl_map(action_kls, action_probs, env, figure_sizes, noshow = True, directory = directory)
  if lso is not None:
    print('')
    print('-'*k+'LSOS'+'-'*k)
    plot_lso_map(lso, action_probs, env, figure_sizes, noshow = True, directory = directory)
    print('')
    print('-'*k+'STATE DENSITIES'+'-'*k)
    plot_state_densities(stats.state_goal_counts, action_probs, env, figure_sizes, noshow = True, directory = directory)
  print('')
  print('-'*k+'POLICY'+'-'*k)
  print_policy(action_probs, env)
  print('')
  print('FINISHED')
  
  return steps_per_reward, action_info, state_info, experiment_name

if __name__ == "__main__":
  train_alice()