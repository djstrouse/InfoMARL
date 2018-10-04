import math
import numpy as np
import itertools
from collections import namedtuple
from play_episode import play

EpisodeStats = namedtuple('Stats', ['episode_lengths',
                                    'episode_rewards', 'episode_modified_rewards',
                                    'episode_lso', 'episode_action_kl',
                                    'episode_keys', 'state_goal_counts'])
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'state_goal_counts'])

def reinforce(env, agent, training_steps, learning_rate,
              entropy_scale, value_scale, action_info_scale, state_info_scale,
              state_count_discount, state_count_smoothing,
              discount_factor, max_episode_length,
              viz_episode_every = 500, print_updates = False):
  """
  REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
  function approximator using policy gradient.
  
  Args:
    env: OpenAI environment.
    agent: policy/value with predict/update functions
    training_steps: number of time steps to train for
    learning_rate: scalar, or vector of length training_steps
    entropy_scale: scalar, or vector of length training_steps
    value_scale: scalar, or vector of length training_steps
    info_scale: scalar, or vector of length training_steps
    discount_factor: time-discount factor
    max_episode_length: max time steps before forced env reset
  
  Returns:
      An EpisodeStats object: see above.
  """
  
  # this allows one to set params to scalars when not wanting to anneal them
  if not isinstance(learning_rate, (list, np.ndarray)):
    learning_rate = [learning_rate]*training_steps
  if not isinstance(entropy_scale, (list, np.ndarray)):
    entropy_scale = [entropy_scale]*training_steps
  if not isinstance(value_scale, (list, np.ndarray)):
    value_scale = [value_scale]*training_steps
  if not isinstance(action_info_scale, (list, np.ndarray)):
    action_info_scale = [action_info_scale]*training_steps
  if not isinstance(state_info_scale, (list, np.ndarray)):
    state_info_scale = [state_info_scale]*training_steps
    
  # flag that tells caller of function whether or not the run had to exit early
  #   due to nans; useful for triggering retraining with new init
  success = True

  # keep track of useful statistics
  episode_lengths = []
  episode_rewards = []
  episode_modified_rewards = []
  if agent.use_action_info:
    episode_action_kl = []
  else:
    episode_action_kl = None
  if agent.use_state_info:
    episode_lso = []
  else:
    episode_lso = None
  if str(env) == 'KeyGame':
    episode_keys = []
  else:
    episode_keys = None
  
  # count state frequencies if using state info
  if agent.use_state_info:
    init_count = 1
    state_goal_counts = init_count * np.ones((env.nS, env.nG))
    state_count_mask = np.zeros((env.nS, env.nS))
    if state_count_smoothing:
      for s in range(env.nS):
        mask = np.zeros(env.nS)
        x,y = env.state_to_coord[s]
        for s2 in range(env.nS):
          x2,y2 = env.state_to_coord[s2]
          d = abs(x-x2)+abs(y-y2) # manhattan distance
          mask[s2] = math.exp(-(d**2)/(state_count_smoothing**2))
        mask *= 1/np.sum(mask) # normalize
        state_count_mask[s,:] = mask
    else:
      state_count_mask = np.eye(env.nS)
    
  else:
    state_goal_counts = None

  # count total steps
  step_count = 0
  last_episode_reward = 0   
  
  # iterate over episodes
  for i in itertools.count(start = 0):
    
    this_learning_rate = learning_rate[step_count]
    this_entropy_scale = entropy_scale[step_count]
    this_value_scale = value_scale[step_count]
    this_action_info_scale = action_info_scale[step_count]
    this_state_info_scale = state_info_scale[step_count]
    
    # occasional viz
    if i % viz_episode_every == 0:
      print('----- EPISODE %i, STEP %i -----\n' % (i, step_count))
      play(env = env,
           alice = agent,
           state_goal_counts = state_goal_counts,
           max_episode_length = max_episode_length)
    
    # Reset the environment and pick the first action
    state, goal = env._reset()
    if agent.use_state_info:
      state_goal_counts *= state_count_discount
      state_goal_counts[:, goal] += state_count_mask[state,:]
    
    episode = []
    episode_length = 0
    total_reward = 0
    total_modified_reward = 0
    if agent.use_action_info: total_action_kl = 0
    else: total_action_kl = None
    if agent.use_state_info: total_lso = 0
    else: total_lso = None
    
    # One step in the environment
    for t in itertools.count(start = 1):
      
      step_count += 1
        
      # Take a step
      action_probs, value, logits = agent.predict(state = state, goal = goal)
      action = np.random.choice(np.arange(len(action_probs)), p = action_probs)
      next_state, reward, done, _ = env.step(action)
      modified_reward = reward # modified reward includes info
      
      # Extract infos, add to totals, add to reward
      if agent.use_action_info:
        kl = agent.get_kl(state = state, goal = goal)
        total_action_kl += kl
        modified_reward += this_action_info_scale * kl
      if agent.use_state_info:
        ps_g = state_goal_counts[state, goal] / np.sum(state_goal_counts[:,goal])
        ps = np.sum(state_goal_counts[state,:]) / np.sum(state_goal_counts)
        lso = np.log2(ps_g/ps)
        total_lso += lso
        modified_reward += this_state_info_scale * lso
        
      # Keep track of the transition (note that modified reward used here since used for learning)
      episode.append(Transition(state = state,
                                action = action,
                                reward = modified_reward,
                                state_goal_counts = state_goal_counts))
        
      total_reward += reward
      total_modified_reward += modified_reward
      episode_length = t
      
      if print_updates:
        # Print out which step we're on, useful for debugging.
        print("\r{}/{} steps, last reward {}, step {} @ episode {}     ".format(
                step_count, training_steps, last_episode_reward, t, i+1), end="")
        # sys.stdout.flush()
          
      state = next_state
      if agent.use_state_info:
        state_goal_counts *= state_count_discount
        state_goal_counts[:, goal] += state_count_mask[state,:]
        
      # check if nans creeped in (to bob's action probabilities)
      if np.isnan(action_probs).any():
        print('NaN alert at %i steps' % step_count)
        print(action_probs)
        print(logits)
        print(state_goal_counts)
        success = False
        break
      
      if done or t > max_episode_length: break
    
    # no more episodes if nans
    if not success: break
    
    # save episode stats
    final_state = next_state
    if str(env) == 'KeyGame': episode_keys.append(env.state_to_key[final_state])
    episode_rewards.append(total_reward)
    episode_modified_rewards.append(total_modified_reward)
    episode_lengths.append(episode_length)
    if agent.use_action_info:
      episode_action_kl.append(total_action_kl)
    if agent.use_state_info:
      episode_lso.append(total_lso)
    last_episode_reward = total_reward

    # go through episode and make agent updates
    for t, transition in enumerate(episode):
      total_return = sum(discount_factor**tau * future.reward for tau, future in enumerate(episode[t:]))
      # if last transition, use next_state from above, since not saved as transition
      if agent.use_state_info:
        if t == len(episode)-1:
          next_state = final_state
        else:
          next_state = episode[t+1].state
      else:
        next_state = None
#      if agent.use_state_info:
#        # calculate log state odds to add to return (HARD CODE N=GAMMA=1)
#        this_goal_count = np.sum(state_goal_counts[:, goal])
#        next_state_count = state_goal_counts[next_state, goal]
#        next_state_prob = next_state_count / this_goal_count # p(s_t|g)
#        np.sum(state_goal_counts[next_state,:])
#        next_total_count = np.sum(state_goal_counts[next_state,:])
#        next_total_prob = next_total_count / np.sum(state_goal_counts) # p(s_t)      
#        next_state_ratio = next_state_prob / next_total_prob
#        total_return += 1 + next_state_ratio
      losses, dstats = agent.update(state = transition.state,
                                    goal = goal,
                                    action = transition.action,
                                    return_estimate = total_return,
                                    learning_rate = this_learning_rate,
                                    entropy_scale = this_entropy_scale,
                                    value_scale = this_value_scale,
                                    action_info_scale = this_action_info_scale,
                                    state_info_scale = this_state_info_scale,
                                    state_goal_counts = transition.state_goal_counts,
                                    next_state = next_state)
      if np.isnan(losses.state_info_loss):
        print(dstats)
        print('state %i' % transition.state)
        print('action %s' % env.index_to_action[transition.action])
        print('next state %i' % next_state)
        print('goal %i' % goal)
        print('return %.1f' % total_return)
                        
    # if exceeded number of steps to train for, quit
    if step_count >= training_steps: break
  
  # package up stats
  stats = EpisodeStats(episode_lengths = episode_lengths,
                       episode_rewards = episode_rewards,
                       episode_modified_rewards = episode_modified_rewards,
                       episode_action_kl = episode_action_kl,
                       episode_lso = episode_lso,
                       episode_keys = episode_keys,
                       state_goal_counts = state_goal_counts)
  
  return stats, success