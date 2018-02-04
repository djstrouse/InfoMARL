import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple

FigureSizes = namedtuple('FigureSizes', ['figure', 'tick_label', 'axis_label', 'title'])

# added dotted lines for Alice!

def plot_multiple_experiments(list_of_directories, list_of_exp_names,
                              figure_sizes, collection_name):
  
  # load results
  results_directory = os.getcwd()+'/results/'
  results = []
  for d in list_of_directories:
    r = pickle.load(open(results_directory+d+'/results.pkl','rb'))
    results.append(r)
    
  # plot them
  fig = plt.figure(figsize = figure_sizes.figure)
  total_steps = []
  for r in results:
    plt.plot(np.cumsum(r.bob.episode_lengths), np.cumsum(r.bob.episode_rewards),
             linewidth = 8)
    total_steps.append(np.sum(r.bob.episode_lengths))
  plt.xlabel("Time Steps", fontsize = figure_sizes.axis_label)
  plt.ylabel("Total Reward", fontsize = figure_sizes.axis_label)
  plt.xlim((0, np.min(total_steps)))
  plt.ylim(ymin = 0)
  plt.title("Total Reward over Time", fontsize = figure_sizes.title)
  plt.legend(list_of_exp_names, loc = 'upper left', fontsize = figure_sizes.axis_label)
  plt.tick_params(labelsize = figure_sizes.tick_label)  
  plt.savefig(os.getcwd()+'/results/'+collection_name+'_reward_per_timestep.eps', format='eps')
  plt.savefig(os.getcwd()+'/results/'+collection_name+'_reward_per_timestep.pdf', format='pdf')
  plt.savefig(os.getcwd()+'/results/'+collection_name+'_reward_per_timestep.png', format='png')
  plt.close(fig)
  
  return
  
    
if __name__ == "__main__":
  figure_sizes = FigureSizes(figure = (50,25),
                             tick_label = 40,
                             axis_label = 50,
                             title = 60)
#  list_of_directories = ['2018_02_03_1449_bob_with_cooperative_alice_3x3',
#                         '2018_02_03_1522_bob_with_ambivalent_alice_3x3',
#                         '2018_02_03_1506_bob_with_competitive_alice_3x3']
#  list_of_exp_names = ['cooperative',
#                       'ambivalent',
#                       'competitive']
#  collection_name = '3x3'
  list_of_directories = ['2018_02_03_2048_bob_with_cooperative_alice_5x5',
                         '2018_02_03_2237_bob_with_competitive_alice_5x5']
  list_of_exp_names = ['cooperative',
                       'competitive']
  collection_name = '5x5'
  fig = plot_multiple_experiments(list_of_directories = list_of_directories,
                                 list_of_exp_names = list_of_exp_names,
                                 figure_sizes = figure_sizes,
                                 collection_name = collection_name)