import numpy as np
import itertools
import copy
from collections import namedtuple

EpisodeStats = namedtuple('Stats', ['episode_lengths', 'episode_rewards', 'episode_kls'])
Transition = namedtuple('Transition', ['state', 'action', 'reward'])

def reinforce(env, policy_estimator, value_estimator, training_steps,
              entropy_scale, beta, discount_factor, max_episode_length):
  """
  REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
  function approximator using policy gradient.
  
  Args:
    env: OpenAI environment.
    policy: policy to be optimized 
    value: value function approximator, used as a baseline
    training_steps: number of time steps to train for
    entropy_scale: vector of length num_episodes
    discount_factor: time-discount factor
  
  Returns:
      An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
  """
  
  # this allows one to set params to scalars when not wanting to anneal them
  if not isinstance(entropy_scale, (list, np.ndarray)):
    entropy_scale = [entropy_scale]*training_steps
  if not isinstance(beta, (list, np.ndarray)):
    beta = [beta]*training_steps

  # Keeps track of useful statistics
  stats = EpisodeStats(episode_lengths = [],
                       episode_rewards = [],
                       episode_kls = []) 

  # count total steps
  step_count = 0
  last_episode_reward = 0   
  
  # iterate over episodes
  for i in itertools.count(start = 0):
    
    this_entropy_scale = entropy_scale[step_count]
    this_beta = beta[step_count]
    
    # Reset the environment and pick the first action
    state, goal = env.reset()
    
    episode = []
    episode_length = 0
    total_reward = 0
    total_kl = 0
    
    # One step in the environment
    for t in itertools.count(start = 1):
      
      step_count += 1
        
      # Take a step
      action_probs = policy_estimator.predict(state, goal)
      kl = policy_estimator.get_kl(state, goal)
      action = np.random.choice(np.arange(len(action_probs)), p = action_probs)
      next_state, reward, done, _ = env.step(action)
      
      # Keep track of the transition
      episode.append(Transition(state = state, action = action, reward = reward))
      
      # Update statistics
      total_reward += reward
      episode_length = t
      total_kl += kl
      
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

    # Go through the episode and make policy updates
    for t, transition in enumerate(episode):
      # The return after this timestep
      total_return = sum(discount_factor**i * t.reward for i, t in enumerate(episode[t:]))
      # Update our value estimator
      value_estimator.update(transition.state, goal, total_return)
      # Calculate baseline/advantage
      baseline_value = value_estimator.predict(transition.state, goal)            
      advantage = total_return - baseline_value
      # Update our policy estimator
      policy_estimator.update(transition.state,
                              goal,
                              advantage,
                              transition.action,
                              this_entropy_scale,
                              this_beta)
      
    # if exceeded number of steps to train for, quit
    if step_count >= training_steps: break
  
  return stats