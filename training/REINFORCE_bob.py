import numpy as np
import itertools
import copy
from collections import namedtuple
from play_episode import play

EpisodeStats = namedtuple('Stats', ['episode_lengths', 'episode_rewards',
                                    'episode_lso', 'episode_action_kl',
                                    'episode_keys', 'state_goal_counts'])
Bobservation = namedtuple('Bobservation', ['alice_states', 'alice_actions', 'state',
                                           'value', 'action', 'reward', 'z'])

def reinforce(env, alice, bob, training_steps, learning_rate,
              entropy_scale, value_scale, discount_factor,
              max_episode_length, state_count_discount = 1, bob_goal_access = None,
              viz_episode_every = 500, print_updates = False):
  """
  REINFORCE (Monte Carlo Policy Gradient) Algorithm for a two-agent system,
  in which the alice is considered part of the environment for bob.
  
  Args:
    env: OpenAI environment.
    alice: a trained agent with a predict function 
    bob: an agent to be trained with predict and update functions
    training_steps: number of time steps to train for
    entropy_scale: vector of length num_episodes, or scalar
    value_scale: vector of length num_episodes, or scalar
    discount_factor: time-discount factor
    max_episode_length: maximum number of time steps for an episode
    bob_goal_access = 'immediate' -> z = +- 1 depending on goal
                    = 'delayed' -> z = +- 1 once alice kl crosses kl_thresh; z = 0 before
                    = None -> z produced by RNN applied to alice trajectory
  
  Returns:
    An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
  """
  
  # this allows one to set params to scalars when not wanting to anneal them
  if not isinstance(learning_rate, (list, np.ndarray)):
    learning_rate = [learning_rate]*training_steps
  if not isinstance(entropy_scale, (list, np.ndarray)):
    entropy_scale = [entropy_scale]*training_steps
  if not isinstance(value_scale, (list, np.ndarray)):
    value_scale = [value_scale]*training_steps

  # flag that tells caller of function whether or not the run had to exit early
  #   due to nans; useful for triggering retraining with new init
  success = True
  
  # each agent needs own copy of env
  if type(env) == tuple: # different versions of env for KeyGame
    alice_env = env[0]
    bob_env = env[1]
  elif str(env) == 'TwoGoalGridWorld':
    alice_env = env
    bob_env = copy.copy(alice_env)

  # Keeps track of useful statistics
  if alice.use_action_info: init_kl = []
  else: init_kl = None
  if alice.use_state_info:
    init_lso = []
    init_count = 1
    init_state_goal_counts = init_count * np.ones((env.nS, env.nG))
  else:
    init_lso = None
    init_state_goal_counts = None
  if str(alice_env) == 'KeyGame':
    episode_keys = []
  else:
    episode_keys = None
  alice_stats = EpisodeStats(episode_lengths = [],
                             episode_rewards = [],
                             episode_keys = episode_keys,
                             episode_action_kl = init_kl,
                             episode_lso = init_lso,
                             state_goal_counts = init_state_goal_counts)
  bob_stats = EpisodeStats(episode_lengths = [],
                           episode_rewards = [],
                           episode_keys = episode_keys,
                           episode_action_kl = None,
                           episode_lso = None,
                           state_goal_counts = None)  
  # count total steps
  step_count = 0
  last_bob_reward = 0
  
  # iterate over episodes
  for i in itertools.count(start = 0):
    
    this_learning_rate = learning_rate[step_count]
    this_entropy_scale = entropy_scale[step_count]
    this_value_scale = value_scale[step_count]
    
    # occasional viz
    if i % viz_episode_every == 0:
      print('----- EPISODE %i, STEP %i -----\n' % (i, step_count))
      play(env = env,
           alice = alice,
           state_goal_counts = alice_stats.state_goal_counts,
           bob = bob,
           max_episode_length = max_episode_length,
           bob_goal_access = bob_goal_access)
  
    # reset envs
    alice_state, goal = alice_env._reset()
    bob_state, _ = bob_env.set_goal(goal)
    if alice.use_state_info:
#      alice_stats.state_goal_counts *= state_count_discount
      alice_stats.state_goal_counts[alice_state, goal] += 1
    
    # initialize alice and bob episode stat trackers
    alice_states = []
    alice_actions = []
    alice_done = False
    alice_episode_length = 0
    alice_total_reward = 0
    if alice.use_action_info: total_action_kl = 0
    else: total_action_kl = None
    if alice.use_state_info: total_lso = 0
    else: total_lso = None
    bob_episode = []
    bob_done = False
    bob_episode_length = 0
    bob_total_reward = 0
    
    # iterate over steps of alice+bob
    for t in itertools.count(start = 1):
      
      # first alice takes a step
      if not alice_done:        
        alice_action_probs, alice_value, alice_logits = alice.predict(alice_state, goal)
        alice_action = np.random.choice(np.arange(len(alice_action_probs)), p = alice_action_probs)
        next_alice_state, alice_reward, alice_done, _ = alice_env.step(alice_action)
        # update alice stats
        alice_total_reward += alice_reward
        alice_episode_length = t
        if alice.use_action_info:
          total_action_kl += alice.get_kl(state = alice_state, goal = goal)
        if alice.use_state_info:
          ps_g = alice_stats.state_goal_counts[alice_state, goal] / np.sum(alice_stats.state_goal_counts[:,goal])
          ps = np.sum(alice_stats.state_goal_counts[alice_state,:]) / np.sum(alice_stats.state_goal_counts)
          total_lso += np.log2(ps_g/ps)
      else: # if done, sit still
        alice_action = alice_env.action_to_index['STAY']
        next_alice_state = alice_state
      alice_states.append(alice_state)
      alice_actions.append(alice_action)
      
      # then bob takes a step
      if not bob_done:
        step_count += 1
        if bob_goal_access is None:
          bob_action_probs, bob_value, z, _ = bob.predict(state = bob_state,
                                                          obs_states = alice_states,
                                                          obs_actions = alice_actions)
        elif bob_goal_access == 'immediate':
          if goal == 0: z = [-1]
          elif goal == 1: z = [+1]
          bob_action_probs, bob_value, _, _ = bob.predict(state = bob_state,
                                                          z = z)
        elif bob_goal_access == 'delayed':
          kl_thresh = .8
          if alice_stats.episode_kls[i]>kl_thresh:
            if goal == 0: z = [-1]
            elif goal == 1: z = [+1]
          else:
            z = [0]
          bob_action_probs, bob_value, _, _ = bob.predict(state = bob_state,
                                                          z = z)
        bob_action = np.random.choice(np.arange(len(bob_action_probs)), p = bob_action_probs)
        next_bob_state, bob_reward, bob_done, _ = bob_env.step(bob_action)
        bob_total_reward += bob_reward
        bob_episode_length = t
        # keep track of the transition for post-episode training
        bob_episode.append(Bobservation(alice_states = alice_states,
                                        alice_actions = alice_actions,
                                        state = bob_state,
                                        value = bob_value,
                                        action = bob_action,
                                        reward = bob_reward,
                                        z = z))
      else: # if done, sit still
        next_bob_state = bob_state
      
      if print_updates:
        # print out which step we're on, useful for debugging.
        print("\r{}/{} steps, last reward {}, step {} @ episode {}     ".format(
                step_count, training_steps, last_bob_reward, t, i+1), end="")
        # sys.stdout.flush()
    
      # check if nans creeped in (to bob's action probabilities)
      if np.isnan(bob_action_probs).any():
        success = False
        break
          
      alice_state = next_alice_state
      bob_state = next_bob_state
      if alice.use_state_info:
#        alice_stats.state_goal_counts *= state_count_discount
        alice_stats.state_goal_counts[alice_state, goal] += 1
      
      # check if episode over
      if (alice_done and bob_done) or t > max_episode_length: break
    
    # no more episodes if nans
    if not success: break
  
    # otherwise, update episode stats
    alice_stats.episode_rewards.append(alice_total_reward)
    alice_stats.episode_lengths.append(alice_episode_length)
    if alice.use_action_info: alice_stats.episode_action_kl.append(total_action_kl)
    if alice.use_state_info: alice_stats.episode_lso.append(total_lso)
    bob_stats.episode_rewards.append(bob_total_reward)
    bob_stats.episode_lengths.append(bob_episode_length)
    last_bob_reward = bob_total_reward
    if str(bob_env) == 'KeyGame':
      alice_stats.episode_keys.append(alice_env.state_to_key[alice_state])
      bob_stats.episode_keys.append(bob_env.state_to_key[bob_state])
  
    # go through the episode and make policy updates
    for t, transition in enumerate(bob_episode):
      total_return = sum(discount_factor**i * t.reward for i, t in enumerate(bob_episode[t:]))
      if bob_goal_access is None: # provide alice trajectory
        bob.update(state = transition.state,
                   action = transition.action,
                   return_estimate = total_return,
                   learning_rate = this_learning_rate,
                   entropy_scale = this_entropy_scale,
                   value_scale = this_value_scale,
                   obs_states = transition.alice_states,
                   obs_actions = transition.alice_actions)
      elif bob_goal_access == 'immediate': # provide static z
        bob.update(state = transition.state,
                   action = transition.action,
                   return_estimate = total_return,
                   learning_rate = this_learning_rate,
                   entropy_scale = this_entropy_scale,
                   value_scale = this_value_scale,
                   z = z)
      elif bob_goal_access == 'delayed': # provide dynamic z
        bob.update(state = transition.state,
                   action = transition.action,
                   return_estimate = total_return,
                   learning_rate = this_learning_rate,
                   entropy_scale = this_entropy_scale,
                   value_scale = this_value_scale,
                   z = transition.z)
    
    # if exceeded number of steps to train for, quit
    if step_count >= training_steps: break
  
  return alice_stats, bob_stats, success