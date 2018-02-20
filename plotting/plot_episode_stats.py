import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util.stats import rate_last_N, first_time_to

def plot_episode_stats(stats, figure_sizes, noshow = False, directory = None):
  
  if type(stats).__name__ == 'Result':
    alice = stats.alice
    stats = stats.bob
    two_agents = True
  else:
    two_agents = False
  
  # Plot the episode length over time (smoothed)
  window = 1000
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
    plt.savefig(directory+'smoothed_episode_lengths.eps', format='eps')
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
    plt.savefig(directory+'episode_lengths.eps', format='eps')
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
    plt.savefig(directory+'episode_rewards.eps', format='eps')
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
    plt.savefig(directory+'reward_per_timestep.eps', format='eps')
    plt.savefig(directory+'reward_per_timestep.pdf', format='pdf')
    plt.savefig(directory+'reward_per_timestep.png', format='png')
  if noshow: plt.close(fig3)
  else: plt.show(fig3)
  
  # Plot episodes per time step
  fig4 = plt.figure(figsize = figure_sizes.figure)
  cumulative_steps = np.cumsum(stats.episode_lengths)
  plt.plot(cumulative_steps, np.arange(len(stats.episode_lengths)), linewidth = 8, label = 'bob')
  if two_agents:
    cumulative_steps = np.cumsum(alice.episode_lengths)
    plt.plot(cumulative_steps, np.arange(len(alice.episode_lengths)), linewidth = 8, label = 'alice')
    plt.legend(loc = 'upper left', fontsize = figure_sizes.axis_label)
  plt.xlabel("Time Steps", fontsize = figure_sizes.axis_label)
  plt.ylabel("Episode", fontsize = figure_sizes.axis_label)
  plt.title("Episodes per time step", fontsize = figure_sizes.title)
  plt.tick_params(labelsize = figure_sizes.tick_label)
  if directory:
    plt.savefig(directory+'episodes_per_timestep.eps', format='eps')
    plt.savefig(directory+'episodes_per_timestep.pdf', format='pdf')
    plt.savefig(directory+'episodes_per_timestep.png', format='png')
  if noshow: plt.close(fig4)
  else: plt.show(fig4)
  
  if stats.episode_kls is not None:
    # Plot a rolling estimate of I(action;goal|state)
    window = 500 # measure in episodes
    fig5 = plt.figure(figsize = figure_sizes.figure)
    info_smoothed = pd.Series(np.asarray(stats.episode_kls)/np.asarray(stats.episode_lengths)).rolling(window, min_periods = window).mean()
    plt.plot(cumulative_steps, info_smoothed)
    plt.xlabel("Time Steps", fontsize = figure_sizes.axis_label)
    plt.ylabel("I(action;goal|state)", fontsize = figure_sizes.axis_label)
    plt.title("Info estimated over sliding window of {} episodes".format(window), fontsize = figure_sizes.title)
    plt.tick_params(labelsize = figure_sizes.tick_label)
    if directory:
      plt.savefig(directory+'info.eps', format='eps')
      plt.savefig(directory+'info.pdf', format='pdf')
      plt.savefig(directory+'info.png', format='png')
    if noshow: plt.close(fig5)
    else: plt.show(fig5)
  else: fig5 = None
  
  # Plot time steps per unit reward (smoothed)
  window = 1000
  fig6 = plt.figure(figsize = figure_sizes.figure)
  total_steps, steps_per_reward = first_time_to(stats.episode_lengths, stats.episode_rewards)
  steps_per_reward_smoothed = pd.Series(steps_per_reward).rolling(window, min_periods = window).mean()
  plt.plot(total_steps, steps_per_reward_smoothed, color = 'b', label = 'bob', linewidth = 8)
  if two_agents:
    average_steps_per_reward = np.sum(alice.episode_lengths)/np.sum(alice.episode_rewards)
    plt.axhline(y = average_steps_per_reward, color = 'r', label = 'alice', linewidth = 8)
    plt.legend(loc = 'upper right', fontsize = figure_sizes.axis_label)
  _, ymax = plt.gca().get_ylim()
  plt.ylim(0, min(6*average_steps_per_reward,ymax))
  plt.xlabel("Time Steps", fontsize = figure_sizes.axis_label)
  plt.ylabel("Time Steps per Reward", fontsize = figure_sizes.axis_label)
  plt.title("Steps per Reward over Time (Smoothed over approximately {} episodes)".format(window), fontsize = figure_sizes.title) 
  plt.tick_params(labelsize = figure_sizes.tick_label)
  if directory:
    plt.savefig(directory+'steps_per_reward.eps', format='eps')
    plt.savefig(directory+'steps_per_reward.pdf', format='pdf')
    plt.savefig(directory+'steps_per_reward.png', format='png')
  if noshow: plt.close(fig6)
  else: plt.show(fig6)

  return fig0, fig1, fig2, fig3, fig4, fig5, fig6