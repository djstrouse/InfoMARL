import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple
os.chdir("..")
from util.stats import rate_last_N, first_time_to

FigureSizes = namedtuple('FigureSizes', ['figure', 'tick_label', 'axis_label', 'title'])

def plot_multiple_experiments(list_of_directories, exp_names_and_colors,
                              figure_sizes, collection_name):
  
  # load results
  results_directory = os.getcwd()+'/results/'
  results = []
  colors = []
  labels = []
  labels_added = set()
  for d in list_of_directories:
    r = pickle.load(open(results_directory+d+'/results.pkl','rb'))
    results.append(r)
    # if directory name contains exp_name, color it with corresponding color
    color_found = False
    for k in exp_names_and_colors.keys():
      if k in d:
        colors.append(exp_names_and_colors[k])
        color_found = True
        if k in labels_added:
          labels.append(None)
        else:
          labels.append(k)
          labels_added.add(k) # will this work?
        break
    if not color_found: raise ValueError('No names in exp_names_and_colors appeared in {}'.format(d))
    
  # plot rewards vs time and write reward rates to text file
  rate_per_what = 100
  f = open(os.getcwd()+'/results/'+collection_name+'_reward_per_timestep.txt','w')
  f.write('REWARD RATES PER %i TIME STEPS\n' % rate_per_what)
  fig1 = plt.figure(figsize = figure_sizes.figure)
  # plot bob
  f.write("***** BOB *****\n")
  for n in range(len(results)):
    r = results[n]
    c = colors[n]
    l = labels[n]
    d = list_of_directories[n]
    cumulative_steps = np.cumsum(r.bob.episode_lengths)
    cumulative_rewards = np.cumsum(r.bob.episode_rewards)
    plt.plot(cumulative_steps, cumulative_rewards,
             color = c, linestyle = '-', label = l, linewidth = 8)
    # write reward rates to text file
    N = 10000
    rate = rate_per_what*rate_last_N(cumulative_steps, cumulative_rewards, N = N)
    f.write("'%s': %i (last %i steps)\n" % (d, rate, N))
  # plot alice
  f.write("***** ALICE *****\n")
  for n in range(len(results)):
    r = results[n]
    c = colors[n]
    l = labels[n]
    d = list_of_directories[n]
    cumulative_steps = np.cumsum(r.alice.episode_lengths)
    cumulative_rewards = np.cumsum(r.alice.episode_rewards)
    plt.plot(cumulative_steps, cumulative_rewards,
             color = c, linestyle = '--', label = None, linewidth = 8)
    # write reward rates to text file
    N = 10000
    rate = rate_per_what*rate_last_N(cumulative_steps, cumulative_rewards, N = N)
    f.write("'%s': %i (last %i steps)\n" % (d, rate, N))
  plt.xlabel("Time Steps", fontsize = figure_sizes.axis_label)
  plt.ylabel("Total Reward", fontsize = figure_sizes.axis_label)
  #plt.xlim((0, np.min(total_steps)))
  #plt.ylim(ymin = 0)
  plt.title("Total Reward over Time", fontsize = figure_sizes.title)
  plt.legend(loc = 'upper left', fontsize = figure_sizes.axis_label)
  plt.tick_params(labelsize = figure_sizes.tick_label)  
  plt.savefig(os.getcwd()+'/results/'+collection_name+'_reward_per_timestep.eps', format='eps')
  plt.savefig(os.getcwd()+'/results/'+collection_name+'_reward_per_timestep.pdf', format='pdf')
  plt.savefig(os.getcwd()+'/results/'+collection_name+'_reward_per_timestep.png', format='png')
  plt.close(fig1)
  
  # plot smoothed episode lengths over time
  window = 1000
  fig2 = plt.figure(figsize = figure_sizes.figure)
  # plot bob
  for n in range(len(results)):
    r = results[n]
    c = colors[n]
    l = labels[n]
    d = list_of_directories[n]
    episode_lengths_smoothed = pd.Series(r.bob.episode_lengths).rolling(window, min_periods = window).mean()
    plt.plot(episode_lengths_smoothed,
             color = c, linestyle = '-', label = l, linewidth = 8)
  # plot alice
  for n in range(len(results)):
    r = results[n]
    c = colors[n]
    l = labels[n]
    d = list_of_directories[n]
    average_episode_length = np.mean(r.alice.episode_lengths)
    plt.axhline(y = average_episode_length,
                color = c, linestyle = '--', label = None, linewidth = 8)
  plt.xlabel("Episode", fontsize = figure_sizes.axis_label)
  plt.ylabel("Episode Length", fontsize = figure_sizes.axis_label)
  plt.title("Episode Length over Time (Smoothed over {} episodes)".format(window), fontsize = figure_sizes.title)
  #plt.xlim((0, np.min(total_steps)))
  plt.ylim(ymin = 0)
  plt.legend(loc = 'upper right', fontsize = figure_sizes.axis_label)
  plt.tick_params(labelsize = figure_sizes.tick_label)  
  plt.savefig(os.getcwd()+'/results/'+collection_name+'_smoothed_episode_lengths.eps', format='eps')
  plt.savefig(os.getcwd()+'/results/'+collection_name+'_smoothed_episode_lengths.pdf', format='pdf')
  plt.savefig(os.getcwd()+'/results/'+collection_name+'_smoothed_episode_lengths.png', format='png')
  plt.close(fig2)  
  
  # Plot time steps per unit reward (smoothed)
  window = 500
  fig3 = plt.figure(figsize = figure_sizes.figure)
  # plot bob
  for n in range(len(results)):
    r = results[n]
    c = colors[n]
    l = labels[n]
    d = list_of_directories[n]
    total_steps, steps_per_reward = first_time_to(r.bob.episode_lengths, r.bob.episode_rewards)
    steps_per_reward_smoothed = pd.Series(steps_per_reward).rolling(window, min_periods = window).mean()
    plt.plot(total_steps, steps_per_reward_smoothed,
             color = c, linestyle = '-', label = l, linewidth = 8)
  # plot alice
  for n in range(len(results)):
    r = results[n]
    c = colors[n]
    l = labels[n]
    d = list_of_directories[n]
    average_steps_per_reward = np.sum(r.alice.episode_lengths)/np.sum(r.alice.episode_rewards)
    plt.axhline(y = average_steps_per_reward,
                color = c, linestyle = '--', label = None, linewidth = 8)
  plt.xlabel("Time Steps", fontsize = figure_sizes.axis_label)
  plt.ylabel("Time Steps per Reward", fontsize = figure_sizes.axis_label)
  plt.title("Steps per Reward over Time (Smoothed over approximately {} episodes)".format(window), fontsize = figure_sizes.title) 
  #plt.xlim((0, np.min(total_steps)))
  _, ymax = plt.gca().get_ylim()
  plt.ylim(0, min(2*average_steps_per_reward,ymax))
  plt.legend(loc = 'upper right', fontsize = figure_sizes.axis_label)
  plt.tick_params(labelsize = figure_sizes.tick_label)  
  plt.savefig(os.getcwd()+'/results/'+collection_name+'_steps_per_reward.eps', format='eps')
  plt.savefig(os.getcwd()+'/results/'+collection_name+'_steps_per_reward.pdf', format='pdf')
  plt.savefig(os.getcwd()+'/results/'+collection_name+'_steps_per_reward.png', format='png')
  plt.close(fig3)

  # Plot time steps per unit reward as % of Alice's
  window = 500
  fig4 = plt.figure(figsize = figure_sizes.figure)
  # plot bob
  for n in range(len(results)-1,-1,-1):
    r = results[n]
    c = colors[n]
    l = labels[n]
    d = list_of_directories[n]
    total_steps, steps_per_reward = first_time_to(r.bob.episode_lengths, r.bob.episode_rewards)
    average_steps_per_reward = np.sum(r.alice.episode_lengths)/np.sum(r.alice.episode_rewards)
    bob_over_alice = steps_per_reward/average_steps_per_reward
    bob_over_alice_smoothed = pd.Series(bob_over_alice).rolling(window, min_periods = window).mean()
    plt.plot(total_steps, bob_over_alice_smoothed,
             color = c, linestyle = '-', label = l, linewidth = 8)
  plt.xlabel("Time Steps", fontsize = figure_sizes.axis_label)
  plt.ylabel("Bob Normalized Episode Length", fontsize = figure_sizes.axis_label)
  plt.title("Bob Steps per Reward / Alice's Average (Smoothed over ~{} episodes)".format(window), fontsize = figure_sizes.title) 
  #plt.xlim((0, np.min(total_steps)))
  #_, ymax = plt.gca().get_ylim()
  plt.ylim((.95, 2))
  plt.legend(loc = 'upper right', fontsize = figure_sizes.axis_label)
  plt.tick_params(labelsize = figure_sizes.tick_label)  
  plt.savefig(os.getcwd()+'/results/'+collection_name+'_normalized_steps_per_reward.eps', format='eps')
  plt.savefig(os.getcwd()+'/results/'+collection_name+'_normalized_steps_per_reward.pdf', format='pdf')
  plt.savefig(os.getcwd()+'/results/'+collection_name+'_normalized_steps_per_reward.png', format='png')
  plt.close(fig4) 
  
  # Plot percentage of time Bob beats Alice to the goal
  window = 1000
  fig5 = plt.figure(figsize = figure_sizes.figure)
  # plot bob
  for n in range(len(results)):
    r = results[n]
    c = colors[n]
    l = labels[n]
    d = list_of_directories[n]
    bob_beats_alice = np.array(r.bob.episode_lengths) < np.array(r.alice.episode_lengths)
    bob_win_percentage = pd.Series(bob_beats_alice).rolling(window, min_periods = window).mean()
    total_steps = np.cumsum(r.bob.episode_lengths)
    plt.plot(total_steps, bob_win_percentage,
             color = c, linestyle = '-', label = l, linewidth = 8)
  plt.xlabel("Time Steps", fontsize = figure_sizes.axis_label)
  plt.ylabel("% of time Bob beats Alice to goal", fontsize = figure_sizes.axis_label)
  plt.title("Bob's Win Percentage (Smoothed over ~{} episodes)".format(window), fontsize = figure_sizes.title) 
  #plt.xlim((0, np.min(total_steps)))
  #_, ymax = plt.gca().get_ylim()
  plt.ylim((0, .5))
  plt.legend(loc = 'upper left', fontsize = figure_sizes.axis_label)
  plt.tick_params(labelsize = figure_sizes.tick_label)  
  plt.savefig(os.getcwd()+'/results/'+collection_name+'_bob_win_percentage.eps', format='eps')
  plt.savefig(os.getcwd()+'/results/'+collection_name+'_bob_win_percentage.pdf', format='pdf')
  plt.savefig(os.getcwd()+'/results/'+collection_name+'_bob_win_percentage.png', format='png')
  plt.close(fig5)
  
  # Plot percentage of time Bob beats or ties Alice to the goal
  window = 1000
  fig6 = plt.figure(figsize = figure_sizes.figure)
  # plot bob
  for n in range(len(results)):
    r = results[n]
    c = colors[n]
    l = labels[n]
    d = list_of_directories[n]
    bob_beats_alice = np.array(r.bob.episode_lengths) <= np.array(r.alice.episode_lengths)
    bob_win_percentage = pd.Series(bob_beats_alice).rolling(window, min_periods = window).mean()
    total_steps = np.cumsum(r.bob.episode_lengths)
    plt.plot(total_steps, bob_win_percentage,
             color = c, linestyle = '-', label = l, linewidth = 8)
  plt.xlabel("Time Steps", fontsize = figure_sizes.axis_label)
  plt.ylabel("% of time Bob beats/ties Alice to goal", fontsize = figure_sizes.axis_label)
  plt.title("Bob's Win+Tie Percentage (Smoothed over ~{} episodes)".format(window), fontsize = figure_sizes.title) 
  #plt.xlim((0, np.min(total_steps)))
  #_, ymax = plt.gca().get_ylim()
  plt.ylim((0, .7))
  plt.legend(loc = 'upper left', fontsize = figure_sizes.axis_label)
  plt.tick_params(labelsize = figure_sizes.tick_label)  
  plt.savefig(os.getcwd()+'/results/'+collection_name+'_bob_win_tie_percentage.eps', format='eps')
  plt.savefig(os.getcwd()+'/results/'+collection_name+'_bob_win_tie_percentage.pdf', format='pdf')
  plt.savefig(os.getcwd()+'/results/'+collection_name+'_bob_win_tie_percentage.png', format='png')
  plt.close(fig6) 
  
  return
  
    
if __name__ == "__main__":
  figure_sizes = FigureSizes(figure = (50,25),
                             tick_label = 40,
                             axis_label = 50,
                             title = 60)
#  list_of_directories = ['job16329146_task1_2018_03_03_175227_bob_with_cooperative_alice_shared128_200k_5x5',
#                         'job16329146_task10_2018_03_03_175618_bob_with_ambivalent_alice_shared128_200k_5x5',
#                         'job16329146_task28_2018_03_03_180353_bob_with_competitive_alice_shared128_200k_5x5',
#                         'job16329146_task35_2018_03_03_175949_bob_with_cooperative_alice_shared128_200k_5x5',
#                         'job16329146_task43_2018_03_03_180523_bob_with_ambivalent_alice_shared128_200k_5x5',
#                         'job16329146_task54_2018_03_03_180637_bob_with_competitive_alice_shared128_200k_5x5',
#                         'job16329146_task61_2018_03_03_181125_bob_with_cooperative_alice_shared128_200k_5x5',
#                         'job16329146_task72_2018_03_03_181107_bob_with_ambivalent_alice_shared128_200k_5x5',
#                         'job16329146_task80_2018_03_03_181246_bob_with_competitive_alice_shared128_200k_5x5',
#                         'job16329146_task99_2018_03_03_181504_bob_with_cooperative_alice_shared128_200k_5x5',
#                         'job16329146_task108_2018_03_03_181827_bob_with_ambivalent_alice_shared128_200k_5x5',
#                         'job16329146_task116_2018_03_03_181731_bob_with_competitive_alice_shared128_200k_5x5',
#                         'job16329146_task126_2018_03_03_181546_bob_with_cooperative_alice_shared128_200k_5x5',
#                         'job16329146_task137_2018_03_03_181458_bob_with_ambivalent_alice_shared128_200k_5x5']
#  list_of_directories = ['job16332603_task5_2018_03_03_211828_bob_with_cooperative_alice_shared128_200k_5x5',
#                         'job16332603_task10_2018_03_03_212312_bob_with_ambivalent_alice_shared128_200k_5x5',
#                         'job16332603_task23_2018_03_03_211523_bob_with_competitive_alice_shared128_200k_5x5',
#                         'job16332603_task35_2018_03_03_211631_bob_with_cooperative_alice_shared128_200k_5x5',
#                         'job16332603_task49_2018_03_03_211433_bob_with_ambivalent_alice_shared128_200k_5x5',
#                         'job16332603_task55_2018_03_03_211915_bob_with_competitive_alice_shared128_200k_5x5',
#                         'job16332603_task69_2018_03_03_211400_bob_with_cooperative_alice_shared128_200k_5x5',
#                         'job16332603_task77_2018_03_03_211647_bob_with_ambivalent_alice_shared128_200k_5x5',
#                         'job16332603_task85_2018_03_03_211755_bob_with_competitive_alice_shared128_200k_5x5',
#                         'job16332603_task96_2018_03_03_211543_bob_with_cooperative_alice_shared128_200k_5x5',
#                         'job16332603_task104_2018_03_03_211550_bob_with_ambivalent_alice_shared128_200k_5x5',
#                         'job16332603_task118_2018_03_03_211436_bob_with_competitive_alice_shared128_200k_5x5',
#                         'job16332603_task123_2018_03_03_211520_bob_with_cooperative_alice_shared128_200k_5x5',
#                         'job16332603_task135_2018_03_03_211653_bob_with_ambivalent_alice_shared128_200k_5x5',
#                         'job16332603_task140_2018_03_03_211811_bob_with_competitive_alice_shared128_200k_5x5']
  list_of_directories = ['job17355731_task2_2018_05_01_192933_bob_with_competitive_state_alice_shared128_200k_5x5',
                         'job17355731_task10_2018_05_01_192922_bob_with_competitive_state_alice_shared128_200k_5x5',
                         'job17355731_task23_2018_05_01_192510_bob_with_competitive_state_alice_shared128_200k_5x5',
                         'job17355731_task32_2018_05_01_192956_bob_with_competitive_state_alice_shared128_200k_5x5',
                         'job17355731_task40_2018_05_01_192944_bob_with_competitive_state_alice_shared128_200k_5x5',
                         'job17355731_task50_2018_05_01_193202_bob_with_cooperative_state_alice_shared128_200k_5x5',
                         'job17355731_task60_2018_05_01_193344_bob_with_cooperative_state_alice_shared128_200k_5x5',
                         'job17355731_task79_2018_05_01_193656_bob_with_cooperative_state_alice_shared128_200k_5x5',
                         'job17355731_task84_2018_05_01_193632_bob_with_cooperative_state_alice_shared128_200k_5x5',
                         'job17355731_task95_2018_05_01_193737_bob_with_cooperative_state_alice_shared128_200k_5x5',
                         'job17355731_task101_2018_05_01_193745_bob_with_ambivalent_state_alice_shared128_200k_5x5',
                         'job17355731_task117_2018_05_01_194253_bob_with_ambivalent_state_alice_shared128_200k_5x5',
                         'job17355731_task121_2018_05_01_194457_bob_with_ambivalent_state_alice_shared128_200k_5x5',
                         'job17355731_task132_2018_05_01_194412_bob_with_ambivalent_state_alice_shared128_200k_5x5',
                         'job17355731_task142_2018_05_01_194631_bob_with_ambivalent_state_alice_shared128_200k_5x5']

  exp_names_and_colors = {'cooperative': 'r',
                          'ambivalent': 'b',
                          'competitive': 'g'}
  collection_name = '5x5_stateinfo_shared128_200k_bestof10'
  fig = plot_multiple_experiments(list_of_directories = list_of_directories,
                                 exp_names_and_colors = exp_names_and_colors,
                                 figure_sizes = figure_sizes,
                                 collection_name = collection_name)