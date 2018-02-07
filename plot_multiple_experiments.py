import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple
from util.stats import rate_last_N

FigureSizes = namedtuple('FigureSizes', ['figure', 'tick_label', 'axis_label', 'title'])

# added dotted lines for Alice!

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
    
  # plot them and write reward rates to text file
  rate_per_what = 100
  f = open(os.getcwd()+'/results/'+collection_name+'_reward_per_timestep.txt','w')
  f.write('REWARD RATES PER %i TIME STEPS\n' % rate_per_what)
  fig = plt.figure(figsize = figure_sizes.figure)
  total_steps = []
  for n in range(len(results)):
    r = results[n]
    c = colors[n]
    l = labels[n]
    d = list_of_directories[n]
    cumulative_steps = np.cumsum(r.bob.episode_lengths)
    cumulative_rewards = np.cumsum(r.bob.episode_rewards)
    plt.plot(cumulative_steps, cumulative_rewards,
             color = c, label = l, linewidth = 8)
    total_steps.append(np.sum(r.bob.episode_lengths))
    # write reward rates to text file
    N1 = 1000
    N2 = 10000
    N3 = 25000
    rate1 = rate_per_what*rate_last_N(cumulative_steps, cumulative_rewards, N = N1)
    rate2 = rate_per_what*rate_last_N(cumulative_steps, cumulative_rewards, N = N2)
    rate3 = rate_per_what*rate_last_N(cumulative_steps, cumulative_rewards, N = N3)
    f.write("'%s': %i (last %i steps), %i (last %i steps) %i (last %i steps)\n"
            % (d, rate1, N1, rate2, N2, rate3, N3))
    
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
  plt.close(fig)
  
  f = open(os.getcwd()+'/results/'+collection_name+'_reward_per_timestep.txt','w')
  f.write("{}: experiment '{}' failed and reran".format(d, experiment_name))
  f.close()
  
  return
  
    
if __name__ == "__main__":
  figure_sizes = FigureSizes(figure = (50,25),
                             tick_label = 40,
                             axis_label = 50,
                             title = 60)
#  list_of_directories = ['2018_02_06_0018_bob_with_cooperative_alice_32_3x3',
#                         '2018_02_06_0103_bob_with_ambivalent_alice_32_3x3',
#                         '2018_02_06_0041_bob_with_competitive_alice_32_3x3',
#                         '2018_02_06_0123_bob_with_cooperative_alice_32_3x3',
#                         '2018_02_06_0146_bob_with_competitive_alice_32_3x3',
#                         '2018_02_06_0207_bob_with_ambivalent_alice_32_3x3',
#                         '2018_02_06_0228_bob_with_cooperative_alice_32_3x3',
#                         '2018_02_06_0252_bob_with_competitive_alice_32_3x3',
#                         '2018_02_06_0317_bob_with_ambivalent_alice_32_3x3']
#  list_of_directories = ['2018_02_06_1255_bob_with_cooperative_alice_64_3x3',
#                         '2018_02_06_1336_bob_with_ambivalent_alice_64_3x3',
#                         '2018_02_06_1316_bob_with_competitive_alice_64_3x3',
#                         '2018_02_06_1354_bob_with_cooperative_alice_64_3x3',
#                         '2018_02_06_1425_bob_with_competitive_alice_64_3x3',
#                         '2018_02_06_1446_bob_with_ambivalent_alice_64_3x3',
#                         '2018_02_06_1507_bob_with_cooperative_alice_64_3x3',
#                         '2018_02_06_1528_bob_with_competitive_alice_64_3x3',
#                         '2018_02_06_1548_bob_with_ambivalent_alice_64_3x3']
#  list_of_directories = ['2018_02_06_1847_bob_with_cooperative_alice_delayed_goal_64_3x3',
#                         '2018_02_06_1854_bob_with_ambivalent_alice_delayed_goal_64_3x3',
#                         '2018_02_06_1900_bob_with_competitive_alice_delayed_goal_64_3x3',
#                         '2018_02_06_2003_bob_with_cooperative_alice_delayed_goal_64_3x3',
#                         '2018_02_06_2009_bob_with_ambivalent_alice_delayed_goal_64_3x3',
#                         '2018_02_06_2016_bob_with_competitive_alice_delayed_goal_64_3x3',
#                         '2018_02_06_2022_bob_with_cooperative_alice_delayed_goal_64_3x3',
#                         '2018_02_06_2028_bob_with_ambivalent_alice_delayed_goal_64_3x3',
#                         '2018_02_06_2035_bob_with_competitive_alice_delayed_goal_64_3x3']
  list_of_directories = ['2018_02_07_0150_bob_with_cooperative_alice_64_200k_3x3',
                         '2018_02_07_0330_bob_with_ambivalent_alice_64_200k_3x3',
                         '2018_02_07_1408_bob_with_competitive_alice_64_200k_3x3']
  exp_names_and_colors = {'cooperative': 'r',
                          'ambivalent': 'b',
                          'competitive': 'g'}
  collection_name = '3x3_64_200k'
#  list_of_directories = ['2018_02_03_2048_bob_with_cooperative_alice_5x5',
#                         '2018_02_03_2237_bob_with_competitive_alice_5x5']
#  collection_name = '5x5'
  fig = plot_multiple_experiments(list_of_directories = list_of_directories,
                                 exp_names_and_colors = exp_names_and_colors,
                                 figure_sizes = figure_sizes,
                                 collection_name = collection_name)