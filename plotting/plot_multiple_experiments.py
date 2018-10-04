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
#    total_steps, steps_per_reward = first_time_to(r.bob.episode_lengths, r.bob.episode_rewards)
    total_steps = r.bob.total_steps
    steps_per_reward = r.bob.steps_per_reward
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
#  plt.title("Steps per Reward over Time (Smoothed over approximately {} episodes)".format(window), fontsize = figure_sizes.title) 
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
#  plt.title("Bob Steps per Reward / Alice's Average (Smoothed over ~{} episodes)".format(window), fontsize = figure_sizes.title) 
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
    bob_beats_alice[np.array(r.bob.episode_rewards)<0] = 0 # filter out episodes where bob goes to wrong goal
    bob_win_percentage = pd.Series(bob_beats_alice).rolling(window, min_periods = window).mean()
    total_steps = np.cumsum(r.bob.episode_lengths)
    plt.plot(total_steps, bob_win_percentage,
             color = c, linestyle = '-', label = l, linewidth = 8)
  plt.xlabel("Time Steps", fontsize = figure_sizes.axis_label)
  plt.ylabel("% of time Bob beats Alice to goal", fontsize = figure_sizes.axis_label)
#  plt.title("Bob's Win Percentage (Smoothed over ~{} episodes)".format(window), fontsize = figure_sizes.title) 
  #plt.xlim((0, np.min(total_steps)))
  #_, ymax = plt.gca().get_ylim()
  plt.ylim((0, 1))
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
    bob_beats_alice[np.array(r.bob.episode_rewards)<0] = 0 # filter out episodes where bob goes to wrong goal
    bob_win_percentage = pd.Series(bob_beats_alice).rolling(window, min_periods = window).mean()
    total_steps = np.cumsum(r.bob.episode_lengths)
    plt.plot(total_steps, bob_win_percentage,
             color = c, linestyle = '-', label = l, linewidth = 8)
  plt.xlabel("Time Steps", fontsize = figure_sizes.axis_label)
  plt.ylabel("% of time Bob beats/ties Alice to goal", fontsize = figure_sizes.axis_label)
#  plt.title("Bob's Win+Tie Percentage (Smoothed over ~{} episodes)".format(window), fontsize = figure_sizes.title) 
  #plt.xlim((0, np.min(total_steps)))
  #_, ymax = plt.gca().get_ylim()
  plt.ylim((0, 1))
  plt.legend(loc = 'upper left', fontsize = figure_sizes.axis_label)
  plt.tick_params(labelsize = figure_sizes.tick_label)  
  plt.savefig(os.getcwd()+'/results/'+collection_name+'_bob_win_tie_percentage.eps', format='eps')
  plt.savefig(os.getcwd()+'/results/'+collection_name+'_bob_win_tie_percentage.pdf', format='pdf')
  plt.savefig(os.getcwd()+'/results/'+collection_name+'_bob_win_tie_percentage.png', format='png')
  plt.close(fig6) 
  
  return
  
    
if __name__ == "__main__":
  figure_sizes = FigureSizes(figure = (50,25),
                             tick_label = 60,
                             axis_label = 80,
                             title = 80)
#  collection_name = '5x5_actioninfo_200k_bestof10'
#  list_of_directories = ['job17566680_task1_2018_05_16_151657_bob_with_cooperative_action_alice_200k_5x5',
#                         'job17566680_task14_2018_05_16_152001_bob_with_cooperative_action_alice_200k_5x5',
#                         'job17566680_task27_2018_05_16_151939_bob_with_cooperative_action_alice_200k_5x5',
#                         'job17566680_task31_2018_05_16_151924_bob_with_cooperative_action_alice_200k_5x5',
#                         'job17566680_task40_2018_05_16_152223_bob_with_cooperative_action_alice_200k_5x5',
#                         'job17566680_task104_2018_05_16_152945_bob_with_ambivalent_action_alice_200k_5x5',
#                         'job17566680_task116_2018_05_16_153315_bob_with_ambivalent_action_alice_200k_5x5',
#                         'job17566680_task122_2018_05_16_153350_bob_with_ambivalent_action_alice_200k_5x5',
#                         'job17566680_task133_2018_05_16_153843_bob_with_ambivalent_action_alice_200k_5x5',
#                         'job17566680_task143_2018_05_16_153701_bob_with_ambivalent_action_alice_200k_5x5',
#                         'job17566680_task54_2018_05_16_152719_bob_with_competitive_action_alice_200k_5x5',
#                         'job17566680_task66_2018_05_16_152320_bob_with_competitive_action_alice_200k_5x5',
#                         'job17566680_task75_2018_05_16_152618_bob_with_competitive_action_alice_200k_5x5',
#                         'job17566680_task83_2018_05_16_152834_bob_with_competitive_action_alice_200k_5x5',
#                         'job17566680_task92_2018_05_16_152923_bob_with_competitive_action_alice_200k_5x5']
#  collection_name = '5x5_stateinfo_200k_bestof10'
#  list_of_directories = ['job17556031_task0_2018_05_16_012805_bob_with_cooperative_state_alice_200k_5x5',
#                         'job17556031_task10_2018_05_16_012702_bob_with_cooperative_state_alice_200k_5x5',
#                         'job17556031_task20_2018_05_16_013141_bob_with_cooperative_state_alice_200k_5x5',
#                         'job17556031_task30_2018_05_16_013110_bob_with_cooperative_state_alice_200k_5x5',
#                         'job17556031_task40_2018_05_16_013010_bob_with_cooperative_state_alice_200k_5x5',
#                         'job17556031_task102_2018_05_16_014647_bob_with_ambivalent_state_alice_200k_5x5',
#                         'job17556031_task113_2018_05_16_014301_bob_with_ambivalent_state_alice_200k_5x5',
#                         'job17556031_task129_2018_05_16_014934_bob_with_ambivalent_state_alice_200k_5x5',
#                         'job17556031_task133_2018_05_16_015016_bob_with_ambivalent_state_alice_200k_5x5',
#                         'job17556031_task140_2018_05_16_014802_bob_with_ambivalent_state_alice_200k_5x5',
#                         'job17556031_task55_2018_05_16_013655_bob_with_competitive_state_alice_200k_5x5',
#                         'job17556031_task63_2018_05_16_013841_bob_with_competitive_state_alice_200k_5x5',
#                         'job17556031_task74_2018_05_16_013701_bob_with_competitive_state_alice_200k_5x5',
#                         'job17556031_task81_2018_05_16_014236_bob_with_competitive_state_alice_200k_5x5',
#                         'job17556031_task96_2018_05_16_014304_bob_with_competitive_state_alice_200k_5x5']
  
#  collection_name = 'keygame_actioninfo_beta.2_discount.8_200k_bestof10'
#  list_of_directories = ['job17587428_task50_2018_05_17_215722_bob_with_cooperative_action_alice_discount0.8_200k_KeyGame',
#                         'job17587428_task60_2018_05_17_215909_bob_with_cooperative_action_alice_discount0.8_200k_KeyGame',
#                         'job17587428_task71_2018_05_17_220115_bob_with_cooperative_action_alice_discount0.8_200k_KeyGame',
#                         'job17587428_task80_2018_05_17_215810_bob_with_cooperative_action_alice_discount0.8_200k_KeyGame',
#                         'job17587428_task90_2018_05_17_220433_bob_with_cooperative_action_alice_discount0.8_200k_KeyGame',
#                         'job17587428_task100_2018_05_17_220219_bob_with_ambivalent_action_alice_discount0.8_200k_KeyGame',
#                         'job17587428_task109_2018_05_17_220842_bob_with_ambivalent_action_alice_discount0.8_200k_KeyGame',
#                         'job17587428_task120_2018_05_17_220831_bob_with_ambivalent_action_alice_discount0.8_200k_KeyGame',
#                         'job17587428_task130_2018_05_17_221559_bob_with_ambivalent_action_alice_discount0.8_200k_KeyGame',
#                         'job17587428_task144_2018_05_17_221123_bob_with_ambivalent_action_alice_discount0.8_200k_KeyGame',
#                         'job17587428_task1_2018_05_17_223345_bob_with_competitive_action_alice_discount0.8_200k_KeyGame',
#                         'job17587428_task11_2018_05_17_220950_bob_with_competitive_action_alice_discount0.8_200k_KeyGame',
#                         'job17587428_task25_2018_05_17_220903_bob_with_competitive_action_alice_discount0.8_200k_KeyGame',
#                         'job17587428_task36_2018_05_17_221015_bob_with_competitive_action_alice_discount0.8_200k_KeyGame',
#                         'job17587428_task40_2018_05_17_220210_bob_with_competitive_action_alice_discount0.8_200k_KeyGame']

  collection_name = 'keygame_actioninfo_beta.25_discount.8_200k_bestof10'
  list_of_directories = ['job17587428_task350_2018_05_17_231018_bob_with_cooperative_state_alice_discount0.8_200k_KeyGame',
                         'job17587428_task360_2018_05_17_231035_bob_with_cooperative_state_alice_discount0.8_200k_KeyGame',
                         'job17587428_task370_2018_05_17_231619_bob_with_cooperative_state_alice_discount0.8_200k_KeyGame',
                         'job17587428_task380_2018_05_17_231855_bob_with_cooperative_state_alice_discount0.8_200k_KeyGame',
                         'job17587428_task390_2018_05_17_231730_bob_with_cooperative_state_alice_discount0.8_200k_KeyGame',
                         'job17587428_task400_2018_05_17_232402_bob_with_ambivalent_state_alice_discount0.8_200k_KeyGame',
                         'job17587428_task410_2018_05_17_232046_bob_with_ambivalent_state_alice_discount0.8_200k_KeyGame',
                         'job17587428_task420_2018_05_17_232249_bob_with_ambivalent_state_alice_discount0.8_200k_KeyGame',
                         'job17587428_task430_2018_05_17_232421_bob_with_ambivalent_state_alice_discount0.8_200k_KeyGame',
                         'job17587428_task440_2018_05_17_232741_bob_with_ambivalent_state_alice_discount0.8_200k_KeyGame',
                         'job17587428_task309_2018_05_17_224836_bob_with_competitive_action_alice_discount0.8_200k_KeyGame',
                         'job17587428_task317_2018_05_17_224941_bob_with_competitive_action_alice_discount0.8_200k_KeyGame',
                         'job17587428_task320_2018_05_17_225252_bob_with_competitive_action_alice_discount0.8_200k_KeyGame',
                         'job17587428_task335_2018_05_17_231437_bob_with_competitive_action_alice_discount0.8_200k_KeyGame',
                         'job17587428_task348_2018_05_17_230533_bob_with_competitive_action_alice_discount0.8_200k_KeyGame']


  exp_names_and_colors = {'cooperative': 'r',
                          'ambivalent': 'b',
                          'competitive': 'g'}
  fig = plot_multiple_experiments(list_of_directories = list_of_directories,
                                 exp_names_and_colors = exp_names_and_colors,
                                 figure_sizes = figure_sizes,
                                 collection_name = collection_name)