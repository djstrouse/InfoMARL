import numpy as np
import sys
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
STAY = 4

action_to_index = {'UP': UP, 'RIGHT': RIGHT, 'DOWN': DOWN, 'LEFT': LEFT, 'STAY': STAY}
index_to_action = {v: k for k, v in action_to_index.items()}

# for more control, could subclass the superclass Env in gym/core.py
class TwoGoalGridWorld(discrete.DiscreteEnv):
  """
  You are an agent on an MxN (M = height, N = width) grid and your goal is to
  reach the terminal state at the top left or the top right corner, depending
  on the episode. For example, a 4x5 grid looks as follows:
  +  o  o  o  -
  o  A  o  o  o
  o  o  o  o  o
  o  o  o  o  o
  A is your position, + is a terminal state with positive reward (r_correct),
  and - a terminal state with negative reward (r_incorrect).
  You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3), or
  choose not to move at all (STAY=4).
  Actions going off the edge leave you in your current state, but incur pentaly r_wall.
  Env may also return a random transition with probability p_rand; no r_wall penalty in this case.
  You receive a reward of r_step at each step until you reach a terminal state.
  goal_locs allows for changing the location or number of goals from default.
  goal_dist allows for changing sampling frequency of goals from default.
  If # of goals is more than 2, r_correct applies only to correct goal, and
  r_incorrect applies to all other goals.
  """

  metadata = {'render.modes': ['human', 'ansi']}

  def __init__(self,
               shape = [3,3],
               r_correct = +1,
               r_incorrect = -1,
               r_step = 0.,
               r_wall = 0.,
               p_rand = 0.,
               goal_locs = None,
               goal_dist = None):
    
    if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
        raise ValueError('shape argument must be a list/tuple of length 2')
    self.shape = shape
    
    # should add checks for scalars here
    self.r_correct = r_correct
    self.r_incorrect = r_incorrect
    self.r_step = r_step
    self.r_wall = r_wall
    self.p_rand = p_rand

    self.nS = np.prod(shape)
    self.nA = 5

    self.max_y = shape[0]
    self.max_x = shape[1]

    # maps goal index to state index
    if goal_locs is None: self.goal_locs = [0, self.max_x-1]
    else: self.goal_locs = goal_locs
    self.nG = len(self.goal_locs)
    
    # maps grid coordinates to state index (and back)
    coord_to_state = {}
    s = 0
    for y in range(self.max_y):
      for x in range(self.max_x):
        coord_to_state[(x,y)] = s
        s += 1
    self.coord_to_state = coord_to_state
    self.state_to_coord = {v: k for k, v in coord_to_state.items()}
    
    # maps action names to index (and back)
    self.action_to_index = action_to_index
    self.index_to_action = index_to_action
    
    # maps goal index to grid coordinates (and back)
    goal_to_coord = {}
    for g in range(self.nG):
      s = self.goal_locs[g]
      coord = self.state_to_coord[s]
      goal_to_coord[g] = coord
    self.goal_to_coord = goal_to_coord
    self.coord_to_goal = {v: k for k, v in goal_to_coord.items()}

    # probability over goals for episodes
    if goal_dist is None: self.goal_dist = np.ones(self.nG) / self.nG
    else: self.goal_dist = goal_dist

    if len(self.goal_locs) != len(self.goal_dist):
      raise ValueError('length of goal_dist and goal_locs must be equal')

    # transition matrix built by _reset
    P = None

    # initial state distribution unused, state init handled by _reset
    # set to uniform distribution so that super call below doesn't get angry
    isd = np.array([1/self.nS]*self.nS)

    super(TwoGoalGridWorld, self).__init__(self.nS, self.nA, P, isd)
    
  def _reward(self, s, hit_wall):
    """Return reward for a given state. Assumes a goal g has been chosen.
    Helper function for _reset."""
    
    # if state is correct goal for this episode, correct
    if s == self.goal_locs[self.g]: reward = self.r_correct
    # if state is a possible goal but not for this episode, incorrect
    elif s in self.goal_locs: reward = self.r_incorrect
    # otherwise, stepwise reward
    else: reward = self.r_step
    # if hit wall, add that reward
    if hit_wall: reward += self.r_wall
    
    return reward
  
  def __str__(self):
    return 'TwoGoalGridWorld'
  
  def _reset(self, goal = None):
    """Overwrites inherited reset to: sample goal, build transition matrix,
    initialize state, and return goal."""
    
    # sample goal
    if goal is None:
      self.g = np.random.choice(self.nG, size = None, p = self.goal_dist)
    else:
      self.g = goal
    
    # build transition matrix
    grid = np.arange(self.nS).reshape(self.shape)
    it = np.nditer(grid, flags=['multi_index'])
    P = {}
    while not it.finished:
      s = it.iterindex
      y, x = it.multi_index

      P[s] = {a : [] for a in range(self.nA)}

      is_done = lambda s: s in self.goal_locs

      # edge case of spawning in a terminal state
      if is_done(s):
        reward = self._reward(s, False)
        P[s][UP] = [(1.0, s, reward, True)]
        P[s][RIGHT] = [(1.0, s, reward, True)]
        P[s][DOWN] = [(1.0, s, reward, True)]
        P[s][LEFT] = [(1.0, s, reward, True)]
        P[s][STAY] = [(1.0, s, reward, True)]
      # not a terminal state
      else:
        # UP
        if y == 0:
          ns_up = s
        else:
          ns_up = s - self.max_x
        # RIGHT
        if x == (self.max_x - 1):
          ns_right = s
        else:
          ns_right = s + 1
        # DOWN
        if y == (self.max_y - 1):
          ns_down = s
        else:
          ns_down = s + self.max_x
        # LEFT
        if x == 0:
          ns_left = s
        else:
          ns_left = s - 1
          
        # random transitions
        if self.p_rand > 0:
          rand_trans = [(self.p_rand/self.nA, ns_up, self._reward(ns_up, False), is_done(ns_up)),
                        (self.p_rand/self.nA, ns_right, self._reward(ns_right, False), is_done(ns_right)),
                        (self.p_rand/self.nA, ns_down, self._reward(ns_down, False), is_done(ns_down)),
                        (self.p_rand/self.nA, ns_left, self._reward(ns_left, False), is_done(ns_left)),
                        (self.p_rand/self.nA, s, self._reward(s, False), is_done(s))]
        else: rand_trans = []
        
        # action-dependent transition probabilities, including random transitions
        P[s][UP] = [(1.0 - self.p_rand, ns_up, self._reward(ns_up, ns_up==s), is_done(ns_up))] + rand_trans
        P[s][RIGHT] = [(1.0 - self.p_rand, ns_right, self._reward(ns_right, ns_right==s), is_done(ns_right))] + rand_trans
        P[s][DOWN] = [(1.0 - self.p_rand, ns_down, self._reward(ns_down, ns_down==s), is_done(ns_down))] + rand_trans
        P[s][LEFT] = [(1.0 - self.p_rand, ns_left, self._reward(ns_left, ns_left==s), is_done(ns_left))] + rand_trans
        P[s][STAY] = [(1.0 - self.p_rand, s, self._reward(s, False), is_done(s))] + rand_trans

      it.iternext()
      
    self.P = P
      
    # sample starting state (uniform, but resample if terminal)
    s = np.random.choice(self.nS)
    while is_done(s): s = np.random.choice(self.nS)
    self.s = s

    self.lastaction = None
    return self.s, self.g
  
  def set_goal(self, goal):
    return self._reset(goal)

  def _render(self, mode = 'human', close = False, bob_state = None):
      if close: return

      outfile = StringIO() if mode == 'ansi' else sys.stdout

      grid = np.arange(self.nS).reshape(self.shape)
      it = np.nditer(grid, flags=['multi_index'])
      while not it.finished:
        s = it.iterindex
        y, x = it.multi_index

        if self.s == s:
          output = " A "
        elif (bob_state is not None) and (s == bob_state):
          output = " B "
        elif s == self.goal_locs[self.g]:
          output = " + "
        elif s in self.goal_locs:
          output = " - "
        else:
          output = " o "

        if x == 0:
          output = output.lstrip() 
        if x == self.shape[1] - 1:
          output = output.rstrip()

        outfile.write(output)

        if x == self.shape[1] - 1:
          outfile.write("\n")

        it.iternext()

if __name__ == "__main__":
  
  import getch, time 
  button_delay = 0.2
   
  env = TwoGoalGridWorld()
  print('wasd to move. w = up, a = left, s = down, d = right, q = stay.')
  input_to_index = {'w': action_to_index['UP'],
                    'a': action_to_index['LEFT'],
                    's': action_to_index['DOWN'],
                    'd': action_to_index['RIGHT'],
                    'q': action_to_index['STAY']}
  done = False
  env._render()
  while not done:
    action = input()
    a = input_to_index[action]
    s, r, done, _ = env.step(a)
    print('action: {}, reward: {}'.format(index_to_action[a], r))
    env._render()
    time.sleep(button_delay)