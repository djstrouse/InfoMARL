import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from util.stats import rate_last_N, mean_last_N, first_time_to

FigureSizes = namedtuple('FigureSizes', ['figure', 'tick_label', 'axis_label', 'title'])

def plot_episode_stats(stats, figure_sizes, noshow = False, directory = None):
  
  if type(stats).__name__ == 'Result':
    alice = stats.alice
    stats = stats.bob
    two_agents = True
  else:
    alice = stats
    two_agents = False
  
  # Plot the episode length over time (smoothed)
  window = 500
  fig0 = plt.figure(figsize = figure_sizes.figure)
  episode_lengths_smoothed = pd.Series(stats.episode_lengths).rolling(window, min_periods = window).mean()
  plt.plot(episode_lengths_smoothed, label = 'bob')
  if two_agents:
    episode_lengths_smoothed = pd.Series(alice.episode_lengths).rolling(window, min_periods = window).mean()
    plt.plot(episode_lengths_smoothed, label = 'alice')
    plt.legend(loc = 'upper right', fontsize = figure_sizes.axis_label)
  plt.xlabel("Episode", fontsize = figure_sizes.axis_label)
  plt.ylabel("Episode Length", fontsize = figure_sizes.axis_label)
  plt.ylim(ymin = 0)
  plt.title("Episode Length over Time (Smoothed over {} episodes)".format(window), fontsize = figure_sizes.title)
  plt.tick_params(labelsize = figure_sizes.tick_label)
  if directory:
    plt.savefig(directory+'smoothed_episode_lengths.pdf', format='pdf')
    plt.savefig(directory+'smoothed_episode_lengths.png', format='png')
  if noshow: plt.close(fig0)
  else: plt.show(fig0)

  # Plot the episode length over time
  fig1 = plt.figure(figsize = figure_sizes.figure)
  plt.plot(stats.episode_lengths, label = 'bob')
  if two_agents:
    plt.plot(alice.episode_lengths, label = 'alice')
    plt.legend(loc = 'upper right', fontsize = figure_sizes.axis_label)
  plt.xlabel("Episode", fontsize = figure_sizes.axis_label)
  plt.ylabel("Episode Length", fontsize = figure_sizes.axis_label)
  plt.ylim(ymin = 0)
  plt.title("Episode Length over Time", fontsize = figure_sizes.title)
  plt.tick_params(labelsize = figure_sizes.tick_label)
  if directory:
    plt.savefig(directory+'episode_lengths.pdf', format='pdf')
    plt.savefig(directory+'episode_lengths.png', format='png')
  if noshow: plt.close(fig1)
  else: plt.show(fig1)

  # Plot the episode reward per episode
  window = 10
  fig2 = plt.figure(figsize = figure_sizes.figure)
  rewards_smoothed = pd.Series(stats.episode_rewards).rolling(window, min_periods = window).mean()
  plt.plot(rewards_smoothed, label = 'bob')
  if two_agents:
    rewards_smoothed = pd.Series(alice.episode_rewards).rolling(window, min_periods = window).mean()
    plt.plot(rewards_smoothed, label = 'alice')
    plt.legend(loc = 'lower right', fontsize = figure_sizes.axis_label)
  plt.xlabel("Episode", fontsize = figure_sizes.axis_label)
  plt.ylabel("Episode Reward (Smoothed)", fontsize = figure_sizes.axis_label)
  plt.title("Episode Reward over Time (Smoothed over window size {})".format(window), fontsize = figure_sizes.title)
  plt.tick_params(labelsize = figure_sizes.tick_label)
  if directory:
    plt.savefig(directory+'episode_rewards.pdf', format='pdf')
    plt.savefig(directory+'episode_rewards.png', format='png')
  if noshow: plt.close(fig2)
  else: plt.show(fig2)
  
  # Plot the episode reward per time step
  fig3 = plt.figure(figsize = figure_sizes.figure) 
  rate_per_what = 100
  N = 10000
  cumulative_steps = np.cumsum(stats.episode_lengths)
  cumulative_rewards = np.cumsum(stats.episode_rewards)
  r = rate_per_what*rate_last_N(cumulative_steps, cumulative_rewards, N = N)
  title = 'Reward per %i steps (last %i steps): %i' % (rate_per_what, N, r)
  plt.plot(cumulative_steps, cumulative_rewards, linewidth = 8, label = 'bob')
  if two_agents:
    cumulative_steps = np.cumsum(alice.episode_lengths)
    cumulative_rewards = np.cumsum(alice.episode_rewards)
    r_alice = rate_per_what*rate_last_N(cumulative_steps, cumulative_rewards, N = N)
    title = 'Reward per %i steps (last %i steps): Bob %i, Alice %i' % (rate_per_what, N, r, r_alice)
    plt.plot(cumulative_steps, cumulative_rewards, linewidth = 8, label = 'alice')
    plt.legend(loc = 'upper left', fontsize = figure_sizes.axis_label)
  plt.xlabel("Time Steps", fontsize = figure_sizes.axis_label)
  plt.ylabel("Total Reward", fontsize = figure_sizes.axis_label)
  plt.title(title, fontsize = figure_sizes.title)
  plt.tick_params(labelsize = figure_sizes.tick_label)
  if directory:
    plt.savefig(directory+'reward_per_timestep.pdf', format='pdf')
    plt.savefig(directory+'reward_per_timestep.png', format='png')
  if noshow: plt.close(fig3)
  else: plt.show(fig3)
  
  if not two_agents:
    # Plot the modified episode reward per episode
    window = 10
    fig4 = plt.figure(figsize = figure_sizes.figure)
    modified_rewards_smoothed = pd.Series(stats.episode_modified_rewards).rolling(window, min_periods = window).mean()
    plt.plot(modified_rewards_smoothed, label = 'bob')
    plt.xlabel("Episode", fontsize = figure_sizes.axis_label)
    plt.ylabel("Modified Episode Reward (Smoothed)", fontsize = figure_sizes.axis_label)
    plt.title("Modified Episode Reward over Time (Smoothed over window size {})".format(window), fontsize = figure_sizes.title)
    plt.tick_params(labelsize = figure_sizes.tick_label)
    if directory:
      plt.savefig(directory+'modified_episode_rewards.pdf', format='pdf')
      plt.savefig(directory+'modified_episode_rewards.png', format='png')
    if noshow: plt.close(fig4)
    else: plt.show(fig4)
  
    # Plot the modified episode reward per time step
    fig5 = plt.figure(figsize = figure_sizes.figure) 
    rate_per_what = 100
    N = 10000
    cumulative_steps = np.cumsum(stats.episode_lengths)
    cumulative_modified_rewards = np.cumsum(stats.episode_modified_rewards)
    r = rate_per_what*rate_last_N(cumulative_steps, cumulative_modified_rewards, N = N)
    title = 'Modified reward per %i steps (last %i steps): %i' % (rate_per_what, N, r)
    plt.plot(cumulative_steps, cumulative_modified_rewards, linewidth = 8, label = 'bob')
    plt.xlabel("Time Steps", fontsize = figure_sizes.axis_label)
    plt.ylabel("Total Modified Reward", fontsize = figure_sizes.axis_label)
    plt.title(title, fontsize = figure_sizes.title)
    plt.tick_params(labelsize = figure_sizes.tick_label)
    if directory:
      plt.savefig(directory+'modified_reward_per_timestep.pdf', format='pdf')
      plt.savefig(directory+'modified_reward_per_timestep.png', format='png')
    if noshow: plt.close(fig5)
    else: plt.show(fig5)

  if alice.episode_action_kl is not None:
    # Plot a rolling estimate of I(action;goal|state)
    window = 1000 # measure in episodes
    fig6 = plt.figure(figsize = figure_sizes.figure)
    cumulative_steps = np.cumsum(alice.episode_lengths)
    info_smoothed = pd.Series(np.asarray(alice.episode_action_kl)/np.asarray(alice.episode_lengths)).rolling(window, min_periods = window).mean()
    N = 10000
    action_info = mean_last_N(cumulative_steps, info_smoothed, N = N)
    plt.plot(cumulative_steps, info_smoothed)
    plt.xlabel("Time Steps", fontsize = figure_sizes.axis_label)
    plt.ylabel("I(action;goal|state)", fontsize = figure_sizes.axis_label)
    plt.title("Info estimated over sliding window of {} episodes".format(window), fontsize = figure_sizes.title)
    plt.tick_params(labelsize = figure_sizes.tick_label)
    if directory:
      plt.savefig(directory+'action_info.pdf', format='pdf')
      plt.savefig(directory+'action_info.png', format='png')
    if noshow: plt.close(fig6)
    else: plt.show(fig6)
  else:
    fig6 = None
    action_info = None
  
  if alice.episode_lso is not None:
    # Plot a rolling estimate of I(state;goal)
    window = 1000 # measure in episodes
    fig7 = plt.figure(figsize = figure_sizes.figure)
    cumulative_steps = np.cumsum(alice.episode_lengths)
    info_smoothed = pd.Series(np.asarray(alice.episode_lso)/np.asarray(alice.episode_lengths)).rolling(window, min_periods = window).mean()
    N = 10000
    state_info = mean_last_N(cumulative_steps, info_smoothed, N = N)
    plt.plot(cumulative_steps, info_smoothed)
    plt.xlabel("Time Steps", fontsize = figure_sizes.axis_label)
    plt.ylabel("I(state;goal)", fontsize = figure_sizes.axis_label)
    plt.title("Info estimated over sliding window of {} episodes".format(window), fontsize = figure_sizes.title)
    plt.tick_params(labelsize = figure_sizes.tick_label)
    if directory:
      plt.savefig(directory+'state_info.pdf', format='pdf')
      plt.savefig(directory+'state_info.png', format='png')
    if noshow: plt.close(fig7)
    else: plt.show(fig7)
  else:
    fig7 = None
    state_info = None
  
  # Plot time steps per unit reward (smoothed)
  window = 500
  fig8 = plt.figure(figsize = figure_sizes.figure)
  total_steps, steps_per_reward = first_time_to(stats.episode_lengths, stats.episode_rewards) 
  N = 10000
  average_steps_per_reward = mean_last_N(total_steps, steps_per_reward, N = N)
  steps_per_reward_smoothed = pd.Series(steps_per_reward).rolling(window, min_periods = window).mean()
  if two_agents: lab = 'bob'
  else: lab = 'alice'
  plt.plot(total_steps, steps_per_reward_smoothed, color = 'b', label = lab, linewidth = 8)
  if two_agents:
    average_steps_per_reward_alice = np.sum(alice.episode_lengths)/np.sum(alice.episode_rewards)
    plt.axhline(y = average_steps_per_reward_alice, color = 'r', label = 'alice', linewidth = 8)
    plt.legend(loc = 'upper right', fontsize = figure_sizes.axis_label)
    _, ymax = plt.gca().get_ylim()
    plt.ylim(0, min(6*average_steps_per_reward_alice,ymax))
    tit = "Smoothed over ~%i episodes, Mean (last %i steps): Bob %.1f, Alice %.1f" % (window, N, average_steps_per_reward, average_steps_per_reward_alice)
  else:
    average_steps_per_reward_alice = None
    tit = "Smoothed over ~%i episodes, Mean (last %i steps): %.1f" % (window, N, average_steps_per_reward)
  plt.title(tit, fontsize = figure_sizes.title)
  plt.xlabel("Time Steps", fontsize = figure_sizes.axis_label)
  plt.ylabel("Time Steps per Reward", fontsize = figure_sizes.axis_label)  
  plt.tick_params(labelsize = figure_sizes.tick_label)
  if directory:
    plt.savefig(directory+'steps_per_reward.pdf', format='pdf')
    plt.savefig(directory+'steps_per_reward.png', format='png')
  if noshow: plt.close(fig8)
  else: plt.show(fig8)
  
  # Plot rate of picking up different key types, if key env
  if alice.episode_keys is not None and not two_agents: # not interesting to consider bob's key pickups
    window = 100
    fig9 = plt.figure(figsize = figure_sizes.figure)
    master_smoothed = pd.Series([k == 'master' for k in stats.episode_keys]).rolling(window, min_periods = window).mean()
    specific_smoothed = pd.Series([type(k) == int for k in stats.episode_keys]).rolling(window, min_periods = window).mean()
    cumulative_steps = np.cumsum(alice.episode_lengths)
    plt.plot(cumulative_steps, master_smoothed,  color = 'b', label = 'master key', linewidth = 8)
    plt.plot(cumulative_steps, specific_smoothed,  color = 'r', label = 'goal-specific key', linewidth = 8)
    plt.legend(loc = 'center right', fontsize = figure_sizes.axis_label)
    plt.xlabel("Time Steps", fontsize = figure_sizes.axis_label)
    plt.ylabel("% Episodes Key Type Picked Up", fontsize = figure_sizes.axis_label)
    plt.tick_params(labelsize = figure_sizes.tick_label)
    if directory:
      plt.savefig(directory+'key_pickups.pdf', format='pdf')
      plt.savefig(directory+'key_pickups.png', format='png')
    if noshow: plt.close(fig9)
    else: plt.show(fig9)

  return average_steps_per_reward, average_steps_per_reward_alice, action_info, state_info

if __name__ == "__main__":
  figure_sizes = FigureSizes(figure = (50,25),
                             tick_label = 60,
                             axis_label = 80,
                             title = 80)
  os.chdir("..")
  experiment = 'job17583555_task101_2018_05_17_1630_alice_negative_action_competitive_beta0.2_discount0.8_250k_KeyGame'
  r = pickle.load(open(os.getcwd()+'/results/'+experiment+'/results.pkl','rb'))
  
  d = os.getcwd()+'/results/'+experiment+'/alt/'
  if not os.path.exists(d): os.makedirs(d)
  
  plot_episode_stats(r, figure_sizes, noshow = True, directory = d)