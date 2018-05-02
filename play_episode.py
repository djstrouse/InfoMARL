import itertools
import copy
import os
import pickle
import tensorflow as tf
import numpy as np
from envs.TwoGoalGridWorld import TwoGoalGridWorld
from agents.bob import RNNObserver
from agents.alice import TabularREINFORCE

def play_from_directory(experiment_name):
  
  cwd = os.getcwd()
  directory = cwd+'/results/'+experiment_name+'/'
  os.chdir(directory)
  #sys.path.append('/results/'+experiment_name)
  
  # unpickle results
  results = pickle.load(open(directory+'results.pkl','rb'))
  
  # import configs
  import alice_config
  alice_agent_param, alice_training_param, alice_experiment_name = alice_config.get_config()
  import env_config
  env_param, _ = env_config.get_config()
  import bob_config
  agent_param, training_param, experiment_name, alice_experiment = bob_config.get_config()

  # initialize experiment using configs
  tf.reset_default_graph()
  #global_step = tf.Variable(0, name = "global_step", trainable = False)
  env = TwoGoalGridWorld(shape = env_param.shape,
                         r_correct = env_param.r_correct,
                         r_incorrect = env_param.r_incorrect,
                         r_step = env_param.r_step,
                         r_wall = env_param.r_wall,
                         p_rand = env_param.p_rand,
                         goal_locs = env_param.goal_locs,
                         goal_dist = env_param.goal_dist)
  with tf.variable_scope('alice'):  
    alice = TabularREINFORCE(env = env,
                             use_action_info = alice_agent_param.use_action_info,
                             use_state_info = alice_agent_param.use_state_info)
    #alice_saver = tf.train.Saver()
  with tf.variable_scope('bob'):
    bob = RNNObserver(env = env,
                      shared_layer_sizes = agent_param.shared_layer_sizes,
                      policy_layer_sizes = agent_param.policy_layer_sizes,
                      value_layer_sizes = agent_param.value_layer_sizes,
                      use_RNN = agent_param.use_RNN)
    bob_saver = tf.train.Saver()
     
  # simulate an episode
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #alice_saver.restore(sess, directory+'alice/alice.ckpt')
    bob_saver.restore(sess, directory+'bob/bob.ckpt')
    play(env = env, alice = alice, bob = bob, results = results,
         bob_goal_access = training_param.bob_goal_access,
         gamma = training_param.discount_factor)
    
  os.chdir(cwd)
    
  return

def play(env, alice, bob, state_goal_counts = None,
         max_episode_length = 100, bob_goal_access = None, gamma = None):
  # if alice.use_state_info, need to include her state_goal_counts
  
  alice_env = env
  bob_env = copy.copy(alice_env)
  
  alice_state, goal = alice_env._reset()
  bob_state, _ = bob_env.set_goal(goal)
  
  alice_states = []
  alice_actions = []
  alice_done = False
  alice_total_reward = 0
  alice_episode_length = 0
  if alice.use_action_info: total_kl = 0
  else: total_kl = None
  if alice.use_state_info: total_lso = 0
  else: total_lso = None
  draw_alice = True
  
  bob_done = False
  bob_rewards = []
  bob_total_reward = 0
  bob_episode_length = 0
  draw_bob = True
  
  # draw initial env
  print('')
  alice_env._render(bob_state = bob_state)
  print('')
  
  # one step in the environment
  for t in itertools.count(start = 1):
    
    # first alice
    if not alice_done:
      
      # take a step
      alice_action_probs, alice_value = alice.predict(alice_state, goal)
      alice_action = np.random.choice(np.arange(len(alice_action_probs)), p = alice_action_probs)
      next_alice_state, alice_reward, alice_done, _ = alice_env.step(alice_action)

      # update stats
      if alice.use_action_info:
        total_kl += alice.get_kl(state = alice_state, goal = goal)
      if alice.use_state_info:
        ps_g = state_goal_counts[alice_state, goal] / np.sum(state_goal_counts[:,goal])
        ps = np.sum(state_goal_counts[alice_state,:]) / np.sum(state_goal_counts)
        total_lso += np.log2(ps_g/ps)
        
      alice_total_reward += alice_reward
      alice_episode_length = t
       
    else: # if done, sit still
      alice_action = alice_env.action_to_index['STAY']
      next_alice_state = alice_state
    alice_states.append(alice_state)
    alice_actions.append(alice_action)
    
    # draw env with alice step
    if draw_alice:
      if total_kl is not None: kl_str = ', tot kl = %.2f' % total_kl
      else: kl_str = ''
      if total_lso is not None: lso_str = ', tot lso = %.2f' % total_lso
      else: lso_str = ''
      print('alice step %i: reward = %i%s%s, action: %s' %
            (t, alice_total_reward, kl_str, lso_str, env.index_to_action[alice_action]))
      print('')
      alice_env._render(bob_state = bob_state)
      print('')
      if alice_done: draw_alice = False # only draw alice step first step after done
      
    # then bob takes a step
    if not bob_done:
      if bob_goal_access is None:
        bob_action_probs, bob_value, z, logits = bob.predict(state = bob_state,
                                                             obs_states = alice_states,
                                                             obs_actions = alice_actions)
      elif bob_goal_access == 'immediate':
        if goal == 0: z = [-1]
        elif goal == 1: z = [+1]
        bob_action_probs, bob_value, _, logits = bob.predict(state = bob_state,
                                                             z = z)
      elif bob_goal_access == 'delayed':
        kl_thresh = .8
        if total_kl>kl_thresh:
          if goal == 0: z = [-1]
          elif goal == 1: z = [+1]
        else:
          z = [0]
        bob_action_probs, bob_value, _, logits = bob.predict(state = bob_state,
                                                             z = z)
      bob_action = np.random.choice(np.arange(len(bob_action_probs)), p = bob_action_probs)
      next_bob_state, bob_reward, bob_done, _ = bob_env.step(bob_action)
      bob_total_reward += bob_reward
      bob_rewards.append(bob_reward)
      bob_episode_length = t
    else: # if done, sit still
      next_bob_state = bob_state
    
    # draw env with bob step
    if draw_bob:
      if bob_goal_access is not None: z = z[0]
      print('bob step %i: reward = %i, value = %.2f, rnn latent = %.2f, action: %s' %
            (t, bob_total_reward, bob_value, z, env.index_to_action[bob_action]))
      print('policy: L = %.2f, U = %.2f, R = %.2f, D = %.2f, S = %.2f' %
            (bob_action_probs[env.action_to_index['LEFT']],
             bob_action_probs[env.action_to_index['UP']],
             bob_action_probs[env.action_to_index['RIGHT']],
             bob_action_probs[env.action_to_index['DOWN']],
             bob_action_probs[env.action_to_index['STAY']]))
      print('logits: L = %.2f, U = %.2f, R = %.2f, D = %.2f, S = %.2f' %
            (logits[env.action_to_index['LEFT']],
             logits[env.action_to_index['UP']],
             logits[env.action_to_index['RIGHT']],
             logits[env.action_to_index['DOWN']],
             logits[env.action_to_index['STAY']]))
      print('')
      alice_env._render(bob_state = next_bob_state)
      print('')
      if bob_done: draw_bob = False # only draw bob step first step after done

    if (alice_done and bob_done) or t > max_episode_length: break
        
    alice_state = next_alice_state
    bob_state = next_bob_state

  # print bob's return for each step  
  if gamma is not None:
    returns = [sum(np.array([gamma**i for i in range(len(bob_rewards)-t)])*np.array(bob_rewards[t:])) for t in range(len(bob_rewards))]
    str_returns = ['%.2f' % r for r in returns]
    print('bob returns:', end = ' ')
    for i in range(len(str_returns)):
      print('%i: %s' % (i+1,str_returns[i].lstrip('0')), end = '')
      if i<len(str_returns)-1: print(',', end = ' ')
    print('')
  
if __name__ == "__main__":
  play_from_directory('job16321705_task6_2018_03_03_020524_bob_with_cooperative_alice_shared128_1M_5x5')
  