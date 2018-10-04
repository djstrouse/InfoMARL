import tensorflow as tf
import os
import sys
import pickle
import datetime
import importlib
import glob
import imp
from collections import namedtuple
from shutil import copy
if "../" not in sys.path: sys.path.append("../") 
from envs.TwoGoalGridWorld import TwoGoalGridWorld
from envs.KeyGame import KeyGame
from agents.bob import RNNObserver
from agents.alice import TabularREINFORCE
from training.REINFORCE_bob import reinforce
from plotting.plot_episode_stats import plot_episode_stats
from util.stats import first_time_to

Result = namedtuple('Result', ['alice', 'bob'])
Stats = namedtuple('Stats', ['episode_lengths',
                             'episode_rewards',
                             'episode_keys',
                             'episode_action_kl',
                             'episode_lso',
                             'state_goal_counts',
                             'steps_per_reward',
                             'total_steps'])

def train_bob(bob_config_ext = '', exp_name_ext = '', exp_name_prefix = '',
              results_directory = None):
  
  if results_directory is None: results_directory = os.getcwd()+'/results/'
  
  # import bob
  local_dir = os.getcwd()
  config = importlib.import_module('bob_config'+bob_config_ext)
  agent_param, training_param, experiment_name, alice_experiment = config.get_config()
  print('Imported Bob.')
  
  # import alice
  alice_directory = results_directory+alice_experiment+'/'
  alice_config = imp.load_source('alice_config', alice_directory+'alice_config.py')
  alice_agent_param, alice_training_param, alice_experiment_name = alice_config.get_config()
  print('Imported Alice.')
  
  # import and init env
  alice_env_config = imp.load_source('env_config', alice_directory+'env_config.py')
  env_type, env_param, env_exp_name_ext = alice_env_config.get_config()
  if env_type == 'key': # separately load env param for alice and bob
    alice_env_param = env_param
    bob_env_config = imp.load_source('env_config', local_dir+'/env_config.py')
    _, bob_env_param, _ = bob_env_config.get_config()
  experiment_name = experiment_name + env_exp_name_ext + exp_name_ext
  if env_type == 'grid':
    env = TwoGoalGridWorld(shape = env_param.shape,
                           r_correct = env_param.r_correct,
                           r_incorrect = env_param.r_incorrect,
                           r_step = env_param.r_step,
                           r_wall = env_param.r_wall,
                           p_rand = env_param.p_rand,
                           goal_locs = env_param.goal_locs,
                           goal_dist = env_param.goal_dist)
    alice_env = env
    bob_env = env
  elif env_type == 'key':
    alice_env = KeyGame(shape = alice_env_param.shape,
                        r_correct = alice_env_param.r_correct,
                        r_incorrect = alice_env_param.r_incorrect,
                        r_step = alice_env_param.r_step,
                        r_wall = alice_env_param.r_wall,
                        p_rand = alice_env_param.p_rand,
                        spawn_locs = alice_env_param.spawn_locs,
                        spawn_dist = alice_env_param.spawn_dist,
                        goal_locs = alice_env_param.goal_locs,
                        goal_dist = alice_env_param.goal_dist,
                        key_locs = alice_env_param.key_locs,
                        master_key_locs = alice_env_param.master_key_locs)
    bob_env = KeyGame(shape = bob_env_param.shape,
                      r_correct = bob_env_param.r_correct,
                      r_incorrect = bob_env_param.r_incorrect,
                      r_step = bob_env_param.r_step,
                      r_wall = bob_env_param.r_wall,
                      p_rand = bob_env_param.p_rand,
                      spawn_locs = bob_env_param.spawn_locs,
                      spawn_dist = bob_env_param.spawn_dist,
                      goal_locs = bob_env_param.goal_locs,
                      goal_dist = bob_env_param.goal_dist,
                      key_locs = bob_env_param.key_locs,
                      master_key_locs = bob_env_param.master_key_locs)
  print('Imported environment.')
   
  # run training, and if nans, creep in, train again until they don't
  success = False
  while not success:

    # initialize alice and bob using configs
    tf.reset_default_graph()
    #global_step = tf.Variable(0, name = "global_step", trainable = False)    
    with tf.variable_scope('alice'):
      alice = TabularREINFORCE(env = alice_env,
                               use_action_info = alice_agent_param.use_action_info,
                               use_state_info = alice_agent_param.use_state_info)
      alice_saver = tf.train.Saver()
    with tf.variable_scope('bob'):
      bob = RNNObserver(alice_env = alice_env,
                        bob_env = bob_env,
                        shared_layer_sizes = agent_param.shared_layer_sizes,
                        policy_layer_sizes = agent_param.policy_layer_sizes,
                        value_layer_sizes = agent_param.value_layer_sizes,
                        use_RNN = agent_param.use_RNN)
      saver = tf.train.Saver()
    print('Initialized Alice and Bob.')
  
    # run experiment
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      alice_saver.restore(sess, alice_directory+'alice.ckpt')
      print('Loaded trained Alice.')
      if env_type == 'key': env = (alice_env, bob_env)
      elif env_type == 'grid': env = alice_env
      alice_stats, bob_stats, success = reinforce(env = env,
                                                  alice = alice,
                                                  bob = bob,
                                                  training_steps = training_param.training_steps,
                                                  learning_rate = training_param.learning_rate,
                                                  entropy_scale = training_param.entropy_scale,
                                                  value_scale = training_param.value_scale,
                                                  discount_factor = training_param.discount_factor,
                                                  max_episode_length = training_param.max_episode_length,
                                                  bob_goal_access = training_param.bob_goal_access)
      if success:
        print('Finished training.')
        # save session
        experiment_directory = exp_name_prefix+datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")+'_'+experiment_name+'/'
        directory = results_directory + experiment_directory
        print('Saving results in %s.' % directory)
        if not os.path.exists(directory+'bob/'): os.makedirs(directory+'bob/')
        save_path = saver.save(sess, directory+'bob/bob.ckpt')
        print('Saved bob to %s.' % save_path)
      else:
        print('Unsucessful run - restarting.')
        f = open('error.txt','a')
        d = datetime.datetime.now().strftime("%A, %B %d, %I:%M:%S %p")
        f.write("{}: experiment '{}' failed and reran\n".format(d, exp_name_prefix+experiment_name))
        f.close()
  
  # save experiment stats
  print('Building Alice stats.')
  alice_total_steps, alice_steps_per_reward = first_time_to(alice_stats.episode_lengths,
                                                            alice_stats.episode_rewards)
  a = Stats(episode_lengths = alice_stats.episode_lengths,
            episode_rewards = alice_stats.episode_rewards,
            episode_keys = alice_stats.episode_keys,
            episode_action_kl = alice_stats.episode_action_kl,
            episode_lso = alice_stats.episode_lso,
            state_goal_counts = alice_stats.state_goal_counts,
            steps_per_reward = alice_steps_per_reward,
            total_steps = alice_total_steps)  
  print('Building Bob stats.')
  bob_total_steps, bob_steps_per_reward = first_time_to(bob_stats.episode_lengths,
                                                        bob_stats.episode_rewards)
  b = Stats(episode_lengths = bob_stats.episode_lengths,
            episode_rewards = bob_stats.episode_rewards,
            episode_keys = bob_stats.episode_keys,
            episode_action_kl = None,
            episode_lso = None,
            state_goal_counts = None,
            steps_per_reward = bob_steps_per_reward,
            total_steps = bob_total_steps)
  
  result = Result(alice = a, bob = b)
  if not os.path.exists(directory): os.makedirs(directory)
  with open(directory+'results.pkl', 'wb') as output:
    # copy to locally-defined Stats objects to make pickle happy
    pickle.dump(result, output, pickle.HIGHEST_PROTOCOL)
  print('Saved stats.')
  
  # copy config file to results directory to ensure experiment repeatable
  copy(os.getcwd()+'/bob_config'+bob_config_ext+'.py', directory+'bob_config.py')
  copy(os.getcwd()+'/env_config.py', directory)
  copy(alice_directory+'alice_config.py', directory)
  print('Copied configs.')
  
  # copy alice checkpoint used
  if not os.path.exists(directory+'alice/'): os.makedirs(directory+'alice/')
  for file in glob.glob(alice_directory+'alice.ckpt*'):
    copy(file, directory+'alice/')
  copy(alice_directory+'checkpoint', directory+'alice/')
  print('Copied Alice.')
      
  # plot experiment and save figures
  FigureSizes = namedtuple('FigureSizes', ['figure', 'tick_label', 'axis_label', 'title'])
  figure_sizes = FigureSizes(figure = (50,25),
                             tick_label = 40,
                             axis_label = 50,
                             title = 60)
  avg_steps_per_reward, avg_steps_per_reward_alice, action_info, state_info = plot_episode_stats(result,
                                                                                                 figure_sizes,
                                                                                                 noshow = True,
                                                                                                 directory = directory)
  print('Figures saved.')
  print('\nAll results saved in {}'.format(directory))
  
  return avg_steps_per_reward, avg_steps_per_reward_alice, action_info, state_info, experiment_name

if __name__ == "__main__":
  train_bob()