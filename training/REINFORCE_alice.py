import numpy as np
import itertools
import copy
from collections import namedtuple

EpisodeStats = namedtuple('Stats', ['episode_lengths', 'episode_rewards', 'episode_kls'])
Transition = namedtuple('Transition', ['state', 'action', 'reward'])

def reinforce(env, agent, training_steps, learning_rate,
              entropy_scale, value_scale, info_scale,
              discount_factor, max_episode_length, print_updates = False):
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
  if not isinstance(info_scale, (list, np.ndarray)):
    info_scale = [info_scale]*training_steps

  # Keeps track of useful statistics
  stats = EpisodeStats(episode_lengths = [],
                       episode_rewards = [],
                       episode_kls = []) 

  # count total steps
  step_count = 0
  last_episode_reward = 0   
  
  # iterate over episodes
  for i in itertools.count(start = 0):
    
    this_learning_rate = learning_rate[step_count]
    this_entropy_scale = entropy_scale[step_count]
    this_info_scale = info_scale[step_count]
    this_value_scale = value_scale[step_count]
    
    # Reset the environment and pick the first action
    state, goal = env._reset()
    
    episode = []
    episode_length = 0
    total_reward = 0
    total_kl = 0
    
    # One step in the environment
    for t in itertools.count(start = 1):
      
      step_count += 1
        
      # Take a step
      action_probs, value, kl = agent.predict(state, goal)
      action = np.random.choice(np.arange(len(action_probs)), p = action_probs)
      next_state, reward, done, _ = env.step(action)
      
      # Keep track of the transition
      episode.append(Transition(state = state, action = action, reward = reward))
      
      # Update statistics
      total_reward += reward
      episode_length = t
      total_kl += kl
      
      if print_updates:
        # Print out which step we're on, useful for debugging.
        print("\r{}/{} steps, last reward {}, step {} @ episode {}     ".format(
                step_count, training_steps, last_episode_reward, t, i+1), end="")
        # sys.stdout.flush()

      if done or t > max_episode_length: break
          
      state = next_state
    
    # save episode stats
    stats.episode_rewards.append(total_reward)
    stats.episode_lengths.append(episode_length)
    stats.episode_kls.append(total_kl)
    last_episode_reward = total_reward

    # go through episode and make agent updates
    for t, transition in enumerate(episode):
      total_return = sum(discount_factor**i * t.reward for i, t in enumerate(episode[t:]))
      agent.update(state = transition.state,
                   goal = goal,
                   action = transition.action,
                   return_estimate = total_return,
                   learning_rate = this_learning_rate,
                   entropy_scale = this_entropy_scale,
                   value_scale = this_value_scale,
                   info_scale = this_info_scale)
      
    # if exceeded number of steps to train for, quit
    if step_count >= training_steps: break
  
  return stats