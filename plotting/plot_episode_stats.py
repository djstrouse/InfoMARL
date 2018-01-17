import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_episode_stats(stats, figure_sizes, smoothing_window = 10, noshow = False, directory = None):
    # Plot the episode length over time
    fig1 = plt.figure(figsize = figure_sizes.figure)
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode", fontsize = figure_sizes.axis_label)
    plt.ylabel("Episode Length", fontsize = figure_sizes.axis_label)
    plt.title("Episode Length over Time", fontsize = figure_sizes.title)
    plt.tick_params(labelsize = figure_sizes.tick_label)
    if directory:
      plt.savefig(directory+'episode_lengths.eps', format='eps')
      plt.savefig(directory+'episode_lengths.pdf', format='pdf')
      plt.savefig(directory+'episode_lengths.png', format='png')
    if noshow: plt.close(fig1)
    else: plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize = figure_sizes.figure)
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods = smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode", fontsize = figure_sizes.axis_label)
    plt.ylabel("Episode Reward (Smoothed)", fontsize = figure_sizes.axis_label)
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window), fontsize = figure_sizes.title)
    plt.tick_params(labelsize = figure_sizes.tick_label)
    if directory:
      plt.savefig(directory+'episode_rewards.eps', format='eps')
      plt.savefig(directory+'episode_rewards.pdf', format='pdf')
      plt.savefig(directory+'episode_rewards.png', format='png')
    if noshow: plt.close(fig2)
    else: plt.show(fig2)
    
    # Plot time steps and episode number
    fig3 = plt.figure(figsize = figure_sizes.figure)
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps", fontsize = figure_sizes.axis_label)
    plt.ylabel("Episode", fontsize = figure_sizes.axis_label)
    plt.title("Episode per time step", fontsize = figure_sizes.title)
    plt.tick_params(labelsize = figure_sizes.tick_label)
    if directory:
      plt.savefig(directory+'episodes_per_timestep.eps', format='eps')
      plt.savefig(directory+'episodes_per_timestep.pdf', format='pdf')
      plt.savefig(directory+'episodes_per_timestep.png', format='png')
    if noshow: plt.close(fig3)
    else: plt.show(fig3)
    
    # Plot a rolling estimate of I(action;goal|state)
    smoothing_window = 500
    fig4 = plt.figure(figsize = figure_sizes.figure)
    info_smoothed = pd.Series(stats.episode_kls/stats.episode_lengths).rolling(smoothing_window, min_periods = smoothing_window).mean()
    plt.plot(info_smoothed)
    plt.xlabel("Episode", fontsize = figure_sizes.axis_label)
    plt.ylabel("I(action;goal|state)", fontsize = figure_sizes.axis_label)
    plt.title("Info estimated over sliding window of {} episodes".format(smoothing_window), fontsize = figure_sizes.title)
    plt.tick_params(labelsize = figure_sizes.tick_label)
    if directory:
      plt.savefig(directory+'info.eps', format='eps')
      plt.savefig(directory+'info.pdf', format='pdf')
      plt.savefig(directory+'info.png', format='png')
    if noshow: plt.close(fig4)
    else: plt.show(fig4)

    return fig1, fig2, fig3, fig4