import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle
import imp
import os
from collections import namedtuple
from envs.TwoGoalGridWorld import TwoGoalGridWorld

def build_grid(grid_size, coord_to_state, metrics, wall_value, skip_value, skip_coord = []):
  """Takes state/goal dependent metrics and represents them on a printable grid."""
  grid = wall_value*np.ones((grid_size[1], grid_size[0])) # -1 assigned to wall color
  for coord, state in coord_to_state.items():
    if coord in skip_coord:
      grid[coord[0],coord[1]] = skip_value # assignment to skip_coords
    else:
      grid[coord[0],coord[1]] = metrics[state]
  return np.transpose(grid)

def plot_arrows(action_probs, terminal_states, state_to_coord, grid_size,
                action_to_index, axis):
  """Visualizes grid world policy. Arrows proportional to action prob."""
  # NOTE: quiver uses a different coordinate system than imshow, resulting in
  #       the necessity of a pretty funky coordinate transform to get them to
  #       match up. I don't fully understand *why* the below transform works,
  #       but it does. Could toy with imshow origin param, grid representation,
  #       etc to make this transformation more understandable.
  arrow_ratio = .5
  X = []
  Y = []
  U = []
  V = []
  for action in range(4):
    for state in range(action_probs.shape[0]):
      if state not in terminal_states: # no arrow for goal state
        X.append(state_to_coord[state][0])
        Y.append(state_to_coord[state][1])
        prob = action_probs[state, action]
        if action == action_to_index['UP']:
          U.append(0)
          V.append(+prob)
        elif action == action_to_index['RIGHT']:
          U.append(+prob)
          V.append(0)
        elif action == action_to_index['DOWN']:
          U.append(0)
          V.append(-prob)
        elif action == action_to_index['LEFT']:
          U.append(-prob)
          V.append(0)
        else:
          raise ValueError('action out of range: must be 0-3')
    if action in [action_to_index['LEFT'], action_to_index['RIGHT']]:
      units = 'x'
      scale_units = 'height'
    else:
      units = 'y'
      scale_units = 'width'
    scale = grid_size[0] / arrow_ratio # correct, or max of two grid_size dim?
    axis.quiver(X, Y, U, V, pivot = 'tail', units = units, scale_units = scale_units, scale = scale, width = .05)

def fill_subplot(ax, goal, grid_size, coord_to_state, terminal_states, metrics,
                 max_val, action_probs, action_to_index, colors):
  """Fills in metric/policy viz for particular goal."""
  state_to_coord = {v: k for k, v in coord_to_state.items()}
  cm = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
  grid = build_grid(grid_size, coord_to_state, metrics, wall_value = -max_val,
                    skip_value = -.5*max_val)
  print(grid)
  im = ax.imshow(grid,
                 interpolation = 'none', origin = 'upper', cmap = cm,
                 vmin = -max_val, vmax = max_val)  
  plot_arrows(action_probs, terminal_states, state_to_coord, grid_size, action_to_index, ax)
  ax.set(adjustable = 'box-forced', aspect = 'equal')
  ax.axis('off')
  return im

def plot_value_map(values, action_probs, env, figure_sizes, noshow = False, directory = None):
  """Visualizes state value and policy overlaid, for all goals."""
  max_value = 1
  colors1 = plt.cm.binary(np.linspace(.2, .8, 128))
  colors2 = plt.cm.viridis_r(np.linspace(.2, .8, 128))
  colors = np.vstack((colors1, colors2))
  fig, axes = plt.subplots(ncols = env.nG, figsize = figure_sizes.figure, sharex = True, sharey = True)
  goal = 0
  for ax in axes.flat:
    im = fill_subplot(ax, goal, env.shape, env.coord_to_state, env.goal_locs,
                      values[:,goal], max_value, action_probs[:,goal,:],
                      env.action_to_index, colors)
    goal += 1
  cbar = fig.colorbar(im, ax = axes.ravel().tolist(), shrink = 0.75,
                      boundaries = np.linspace(0, max_value, 101),
                      ticks = np.linspace(0, 1, 5))
  cbar.ax.tick_params(labelsize = figure_sizes.tick_label)
  #plt.suptitle('Value function', fontsize = figure_sizes.title)
  if directory:
      plt.savefig(directory+'values.eps', format='eps')
      plt.savefig(directory+'values.pdf', format='pdf')
      plt.savefig(directory+'values.png', format='png')
  if noshow: plt.close(fig)
  else: plt.show(block = False)
  
def plot_kl_map(kls, action_probs, env, figure_sizes, noshow = False, directory = None):
  """Visualizes kls and policy overlaid, for all goals."""
  max_value = 1
  colors1 = plt.cm.binary(np.linspace(.2, .8, 128))
  colors2 = plt.cm.viridis_r(np.linspace(.2, .8, 128))
  colors = np.vstack((colors1, colors2))
  fig, axes = plt.subplots(ncols = env.nG, figsize = figure_sizes.figure, sharex = True, sharey = True)
  goal = 0
  for ax in axes.flat:
    im = fill_subplot(ax, goal, env.shape, env.coord_to_state, env.goal_locs,
                      kls[:,goal], max_value, action_probs[:,goal,:],
                      env.action_to_index, colors)
    goal += 1
  cbar = fig.colorbar(im, ax = axes.ravel().tolist(), shrink = 0.75,
                      boundaries = np.linspace(0, max_value, 101),
                      ticks = np.linspace(0, 1, 5))
  cbar.ax.tick_params(labelsize = figure_sizes.tick_label)
  #plt.suptitle('I(action;goal|state)', fontsize = figure_sizes.title)
  if directory:
      plt.savefig(directory+'kls.eps', format='eps')
      plt.savefig(directory+'kls.pdf', format='pdf')
      plt.savefig(directory+'kls.png', format='png')
  if noshow: plt.close(fig)
  else: plt.show(block = False)
  
def plot_lso_map(lsos, action_probs, env, figure_sizes, noshow = False, directory = None, ext = ''):
  """Visualizes log state odds and policy overlaid, for all goals."""
  max_value = 1
#  colors 1= plt.cm.binary(np.linspace(.2, .8, 128))
  colors = plt.cm.viridis_r(np.linspace(.2, .8, 128))
#  colors = np.vstack((colors1, colors2))
  fig, axes = plt.subplots(ncols = env.nG, figsize = figure_sizes.figure, sharex = True, sharey = True)
  goal = 0
  for ax in axes.flat:
    im = fill_subplot(ax, goal, env.shape, env.coord_to_state, env.goal_locs,
                      lsos[:,goal], max_value, action_probs[:,goal,:],
                      env.action_to_index, colors)
    goal += 1
  cbar = fig.colorbar(im, ax = axes.ravel().tolist(), shrink = 0.75,
                      boundaries = np.linspace(-max_value, max_value, 101),
                      ticks = np.linspace(-1, 1, 5))
  cbar.ax.tick_params(labelsize = figure_sizes.tick_label)
  #plt.suptitle('I(state;goal)', fontsize = figure_sizes.title)
  if directory:
      plt.savefig(directory+'lsos'+ext+'.eps', format='eps')
      plt.savefig(directory+'lsos'+ext+'.pdf', format='pdf')
      plt.savefig(directory+'lsos'+ext+'.png', format='png')
  if noshow: plt.close(fig)
  else: plt.show(block = False)
  
  # now the absolute values
  colors = np.vstack((colors, colors))
  fig, axes = plt.subplots(ncols = env.nG, figsize = figure_sizes.figure, sharex = True, sharey = True)
  goal = 0
  for ax in axes.flat:
    im = fill_subplot(ax, goal, env.shape, env.coord_to_state, env.goal_locs,
                      abs(lsos[:,goal]), max_value, action_probs[:,goal,:],
                      env.action_to_index, colors)
    goal += 1
  cbar = fig.colorbar(im, ax = axes.ravel().tolist(), shrink = 0.75,
                      boundaries = np.linspace(0, max_value, 101),
                      ticks = np.linspace(0, 1, 5))
  cbar.ax.tick_params(labelsize = figure_sizes.tick_label)
  #plt.suptitle('I(state;goal)', fontsize = figure_sizes.title)
  if directory:
      plt.savefig(directory+'alsos'+ext+'.eps', format='eps')
      plt.savefig(directory+'alsos'+ext+'.pdf', format='pdf')
      plt.savefig(directory+'alsos'+ext+'.png', format='png')
  if noshow: plt.close(fig)
  else: plt.show(block = False)

def plot_state_densities(state_goal_counts, action_probs, env, figure_sizes, noshow = False, directory = None):
  """Visualizes state densities and policy overlaid, for all goals."""
  max_value = 0
  for g in range(state_goal_counts.shape[1]):
    state_densities = state_goal_counts[:,g] / np.sum(state_goal_counts[:,g])
    max_value = max(max_value, max(state_densities))
  colors1 = plt.cm.binary(np.linspace(.2, .8, 128))
  colors2 = plt.cm.viridis_r(np.linspace(.2, .8, 128))
  colors = np.vstack((colors1, colors2))
  fig, axes = plt.subplots(ncols = env.nG, figsize = figure_sizes.figure, sharex = True, sharey = True)
  goal = 0
  for ax in axes.flat:
    state_densities = state_goal_counts[:,goal] / np.sum(state_goal_counts[:,goal])
    im = fill_subplot(ax, goal, env.shape, env.coord_to_state, env.goal_locs,
                      state_densities, max_value, action_probs[:,goal,:],
                      env.action_to_index, colors)
    goal += 1
  cbar = fig.colorbar(im, ax = axes.ravel().tolist(), shrink = 0.75,
                      boundaries = np.linspace(0, max_value, 101),
                      ticks = np.linspace(0, max_value, 5))
  cbar.ax.tick_params(labelsize = figure_sizes.tick_label)
  #plt.suptitle('I(state;goal)', fontsize = figure_sizes.title)
  if directory:
      plt.savefig(directory+'state_densities.eps', format='eps')
      plt.savefig(directory+'state_densities.pdf', format='pdf')
      plt.savefig(directory+'state_densities.png', format='png')
  if noshow: plt.close(fig)
  else: plt.show(block = False)

def print_policy(action_probs, env):
  """Prints state-by-state action probabilities."""
  for g in range(env.nG):
    print('-'*10+'GOAL %i'%g+'-'*10)
    for s in range(env.nS):
      if s not in env.goal_locs:
        print('state %i @ (%i,%i): up = %.2f, down = %.2f, left = %.2f, right = %.2f, stay = %.2f' %
              (s, env.state_to_coord[s][0], env.state_to_coord[s][1],
               action_probs[s,g,env.action_to_index['UP']],
               action_probs[s,g,env.action_to_index['DOWN']],
               action_probs[s,g,env.action_to_index['LEFT']],
               action_probs[s,g,env.action_to_index['RIGHT']],
               action_probs[s,g,env.action_to_index['STAY']]))
      else:
        print('state %i @ (%i,%i): terminal state' %
              (s, env.state_to_coord[s][0], env.state_to_coord[s][1]))
        
        
if __name__ == "__main__":
  
  experiment = 'job17553636_task54_2018_05_15_2220_alice_unregularized_state_ambivalent_250k_5x5'
  
  # load from directory
  os.chdir("..")
  directory = os.getcwd()+'/results/'+experiment+'/'
  r = pickle.load(open(directory+'results.pkl','rb'))
  values = r.values
  action_probs = r.action_probs
  action_kls = r.action_kls
  lso = r.log_state_odds
  state_goal_counts = r.state_goal_counts
  # load env
  env_config = imp.load_source('env_config', directory+'env_config.py')
  env_type, env_param, env_exp_name_ext = env_config.get_config()
  if env_type == 'grid':
    env = TwoGoalGridWorld(shape = env_param.shape,
                           r_correct = env_param.r_correct,
                           r_incorrect = env_param.r_incorrect,
                           r_step = env_param.r_step,
                           r_wall = env_param.r_wall,
                           p_rand = env_param.p_rand,
                           goal_locs = env_param.goal_locs,
                           goal_dist = env_param.goal_dist)
  else:
    raise ValueError('Invalid env.')
  
  # figure sizes
  FigureSizes = namedtuple('FigureSizes', ['figure', 'tick_label', 'axis_label', 'title'])
  figure_sizes = FigureSizes(figure = (50,25),
                             tick_label = 40,
                             axis_label = 50,
                             title = 60)
  # do the plots
  k = 15
  print('-'*k+'VALUES'+'-'*k)
  plot_value_map(values, action_probs, env, figure_sizes, noshow = True, directory = directory)
  if action_kls is not None:
    print('')
    print('-'*k+'KLS'+'-'*k)
    plot_kl_map(action_kls, action_probs, env, figure_sizes, noshow = True, directory = directory)
  if lso is not None:
    print('')
    print('-'*k+'LSOS'+'-'*k)
    plot_lso_map(lso, action_probs, env, figure_sizes, noshow = True, directory = directory)
    print('')
    print('-'*k+'STATE DENSITIES'+'-'*k)
    plot_state_densities(state_goal_counts, action_probs, env, figure_sizes, noshow = True, directory = directory)
  print('')
  print('-'*k+'POLICY'+'-'*k)
  print_policy(action_probs, env)