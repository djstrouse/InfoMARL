import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def build_grid(grid_size, coord_to_state, metrics, wall_value, skip_value, skip_coord = []):
  """Takes state/goal dependent metrics and represents them on a printable grid."""
  grid = wall_value*np.ones((grid_size[1], grid_size[0])) # -1 assigned to wall color
  for coord, state in coord_to_state.items():
    if coord in skip_coord:
      grid[coord[0],coord[1]] = skip_value # assignment to skip_coords
    else:
      grid[coord[0],coord[1]] = metrics[state]
  return np.transpose(grid)

def plot_arrows(action_probs, terminal_states, state_to_coord, grid_size, axis):
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
        if action == UP:
          U.append(0)
          V.append(+prob)
        elif action == RIGHT:
          U.append(+prob)
          V.append(0)
        elif action == DOWN:
          U.append(0)
          V.append(-prob)
        elif action == LEFT:
          U.append(-prob)
          V.append(0)
        else:
          raise ValueError('action out of range: must be 0-3')
    if action in [LEFT, RIGHT]:
      units = 'x'
      scale_units = 'height'
    else:
      units = 'y'
      scale_units = 'width'
    scale = grid_size[0] / arrow_ratio # correct, or max of two grid_size dim?
    axis.quiver(X, Y, U, V, pivot = 'tail', units = units, scale_units = scale_units, scale = scale, width = .05)

def fill_subplot(ax, goal, grid_size, coord_to_state, terminal_states, metrics,
                 max_val, action_probs, colors):
  """Fills in metric/policy viz for particular goal."""
  state_to_coord = {v: k for k, v in coord_to_state.items()}
  cm = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
  grid = build_grid(grid_size, coord_to_state, metrics, wall_value = -max_val,
                    skip_value = -.5*max_val)
  print(grid)
  im = ax.imshow(grid,
                 interpolation = 'none', origin = 'upper', cmap = cm,
                 vmin = -max_val, vmax = max_val)  
  plot_arrows(action_probs, terminal_states, state_to_coord, grid_size, ax)
  ax.set(adjustable = 'box-forced', aspect = 'equal')
  ax.axis('off')
  return im

def plot_value_map(values, action_probs, env):
  """Visualizes state value and policy overlaid, for all goals."""
  max_value = 1
  colors1 = plt.cm.binary(np.linspace(.2, .8, 128))
  colors2 = plt.cm.viridis_r(np.linspace(.2, .8, 128))
  colors = np.vstack((colors1, colors2))
  fig, axes = plt.subplots(ncols = env.nG, sharex = True, sharey = True)
  goal = 0
  for ax in axes.flat:
    im = fill_subplot(ax, goal, env.shape, env.coord_to_state, env.goal_locs,
                      values[:,goal], max_value, action_probs[:,goal,:], colors)
    goal += 1
  cbar = fig.colorbar(im, ax = axes.ravel().tolist(), shrink = 0.95,
                      boundaries = np.linspace(0, max_value, 101),
                      ticks = np.linspace(0, 1, 5))
  plt.show(block = False)
  
def plot_kl_map(kls, action_probs, env):
  """Visualizes kls and policy overlaid, for all goals."""
  max_value = 1
  colors1 = plt.cm.binary(np.linspace(.2, .8, 128))
  colors2 = plt.cm.viridis_r(np.linspace(.2, .8, 128))
  colors = np.vstack((colors1, colors2))
  fig, axes = plt.subplots(ncols = env.nG, sharex = True, sharey = True)
  goal = 0
  for ax in axes.flat:
    im = fill_subplot(ax, goal, env.shape, env.coord_to_state, env.goal_locs,
                      kls[:,goal], max_value, action_probs[:,goal,:], colors)
    goal += 1
  cbar = fig.colorbar(im, ax = axes.ravel().tolist(), shrink = 0.95,
                      boundaries = np.linspace(0, max_value, 101),
                      ticks = np.linspace(0, 1, 5))
  plt.show(block = False)  
  
def print_policy(action_probs, env):
  """Prints state-by-state action probabilities."""
  for g in range(env.nG):
    print('-'*10+'GOAL %i'%g+'-'*10)
    for s in range(env.nS):
      if s not in env.goal_locs:
        print('state %i @ (%i,%i): up = %.2f, down = %.2f, left = %.2f, right = %.2f' %
              (s, env.state_to_coord[s][0], env.state_to_coord[s][1],
               action_probs[s,g,UP], action_probs[s,g,DOWN],
               action_probs[s,g,LEFT], action_probs[s,g,RIGHT]))
      else:
        print('state %i @ (%i,%i): terminal state' %
              (s, env.state_to_coord[s][0], env.state_to_coord[s][1]))