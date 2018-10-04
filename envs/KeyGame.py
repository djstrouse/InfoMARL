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
class KeyGame(discrete.DiscreteEnv):
  """
  You are an agent on an MxN (M = height, N = width) grid and your goal is to
  open the door at the top left or bottom left corner, depending on the episode.
  Each door has one or more keys that specifically open them. There may also
  be "master keys" available that open all doors. The agent can pick up one key
  per episode - whichever key it first steps on to.
  For example, a 3x5 game looks as follows:
  +  o  o  P  o  o
  o  o  o  A  o  U
  -  o  o  M  o  o
  A is agent position, + is a terminal state with positive reward (r_correct),
   - a terminal state with negative reward (r_incorrect), P is the key for the
   + door, M is the key for - door, and U is a master key.
  Agent can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3), or
  choose not to move at all (STAY=4).
  Actions going off the edge leave agent in  current state, but incur pentaly r_wall.
  Env may also return a random transition with probability p_rand; no r_wall penalty in this case.
  Agent receives a reward of r_step at each step until reaching a terminal state.
  spawn_locs allows for changing where the agent spawns.
  (list of state indices, e.g. [9])
  spawn_dist allows for changing the distribution of over spawn_locs.
  (list of probs, e.g. [1.])
  goal_locs allows for changing the location or number of goals from default.
  (list of state indices, e.g. [0,12])
  goal_dist allows for changing sampling frequency of goals from default.
  (list of probs, e.g. [.4, .6])
  If # of goals is more than 2, r_correct applies only to correct goal, and
  r_incorrect applies to all other goals.
  key_locs allows for setting the locations of the door-specific keys.
  (list of lists of key locs of same length as goal locs, e.g. [[1,3],[13,15]])
  master_key_locs allows for setting the locations of the master keys.
  (list of state indices, e.g. [11])
  """

  metadata = {'render.modes': ['human', 'ansi']}

  def __init__(self,
               shape = [3,6],
               r_correct = +1,
               r_incorrect = -1,
               r_step = 0.,
               r_wall = 0.,
               p_rand = 0.,
               spawn_locs = [9],
               spawn_dist = None,
               goal_locs = None,
               goal_dist = None,
               key_locs = [[3],[15]],
               master_key_locs = [11]):
    
    if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
        raise ValueError('shape argument must be a list/tuple of length 2')
    self.shape = shape
    
    # should add checks for scalars here
    self.r_correct = r_correct
    self.r_incorrect = r_incorrect
    self.r_step = r_step
    self.r_wall = r_wall
    self.p_rand = p_rand
    self.spawn_locs = spawn_locs
    self.key_locs = key_locs
    self.master_key_locs = master_key_locs

    self.nL = np.prod(shape) # number of locations
    self.nA = 5 # number of actions

    self.max_y = shape[0]
    self.max_x = shape[1]

    # maps goal index to state index (default = top and bottom left)
    if goal_locs is None: self.goal_locs = [0, int((self.max_y-1)*self.max_x)]
    else: self.goal_locs = goal_locs
    self.nG = len(self.goal_locs)
    
    # maps grid coordinates to loc index (and back)
    coord_to_index = {}
    l = 0
    for y in range(self.max_y):
      for x in range(self.max_x):
        coord_to_index[(x,y)] = l
        l += 1
    self.coord_to_index = coord_to_index
    self.index_to_coord = {v: k for k, v in coord_to_index.items()}
    
    # maps action names to index (and back)
    self.action_to_index = action_to_index
    self.index_to_action = index_to_action
    
    # maps goal index to grid coordinates (and back)
    goal_to_coord = {}
    for g in range(self.nG):
      l = self.goal_locs[g]
      coord = self.index_to_coord[l]
      goal_to_coord[g] = coord
    self.goal_to_coord = goal_to_coord
    self.coord_to_goal = {v: k for k, v in goal_to_coord.items()}

    # probability over goals for episodes
    if goal_dist is None: self.goal_dist = np.ones(self.nG) / self.nG
    else: self.goal_dist = goal_dist
    
    # probability over spawn locs
    if spawn_dist is None: self.spawn_dist = np.ones(len(spawn_locs)) / len(spawn_locs)
    else: self.spawn_dist = spawn_dist

    if len(self.goal_locs) != len(self.goal_dist):
      raise ValueError('length of goal_dist and goal_locs must be equal')
    if len(self.spawn_locs) != len(self.spawn_dist):
      raise ValueError('length of spawn_dist and spawn_locs must be equal')
      
    # construct metastate, which includes possession of key
    self.possession_states = [None, 'master'] + list(range(self.nG)) # none, master, goal-specific
    self.state_to_loc = [] # list
    self.state_to_key = [] # list
    self.loc_and_key_to_state = {} # dictionary
    s = 0
    for p in self.possession_states:
      for l in range(self.nL):
        self.state_to_loc.append(l)
        self.state_to_key.append(p)
        self.loc_and_key_to_state[(l,p)] = s
        s += 1
    self.nS = s
    
    # build dict mapping locations to key indices
    self.loc_to_key = {}
    for g in range(self.nG):
      for l in key_locs[g]:
        if l in self.loc_to_key: raise ValueError('Key collision! Cannot place more than one key in each location.')
        self.loc_to_key[l] = g
    if self.master_key_locs is not None:
      for l in self.master_key_locs:
        if l in self.loc_to_key: raise ValueError('Key collision! Cannot place more than one key in each location.')
        self.loc_to_key[l] = 'master' # nG = master key index

    # transition matrix built by _reset
    P = None

    # initial state distribution unused, state init handled by _reset
    # set to uniform distribution so that super call below doesn't get angry
    isd = np.array([1/self.nS]*self.nS)

    super(KeyGame, self).__init__(self.nS, self.nA, P, isd)
    
  def __str__(self):
    return 'KeyGame'
    
  def _reward(self, l, hit_wall):
    """Return reward for a given loc. Assumes a goal g has been chosen.
    Helper function for _reset."""
    
    # if state is correct goal for this episode, correct
    if l == self.goal_locs[self.g]: reward = self.r_correct
    # if state is a possible goal but not for this episode, incorrect
    elif l in self.goal_locs: reward = self.r_incorrect
    # otherwise, stepwise reward
    else: reward = self.r_step
    # if hit wall, add that reward
    if hit_wall: reward += self.r_wall
    
    return reward
  
  def _update_key(self, p, l):
    """Given new location and current key possession, updates the latter."""
    if p is None and l in self.loc_to_key: # if no key and key in loc
      np = self.loc_to_key[l] # update key possession
    else:
      np = p
    return np
  
  def _locked(self, l, p):
    """Given proposed location and current key possession, checks if door locked."""
    if l in self.goal_locs and p != 'master' and p != self.goal_locs.index(l):
      return True
    else:
      return False
  
  def _reset(self, goal = None):
    """Overwrites inherited reset to: sample goal, build transition matrix,
    initialize state, and return goal."""
    
    # sample goal
    if goal is None:
      self.g = np.random.choice(self.nG, size = None, p = self.goal_dist)
    else:
      self.g = goal
    
    # build transition matrix
    grid = np.arange(self.nL).reshape(self.shape)
    it = np.nditer(grid, flags=['multi_index'])
    P = {}
    
    while not it.finished: # iterate over spatial locations
      
      l = it.iterindex
      y, x = it.multi_index
      
      for p in self.possession_states:

        s = self.loc_and_key_to_state[(l,p)]
  
        P[s] = {a : [] for a in range(self.nA)}
  
        is_done = lambda l: l in self.goal_locs
  
        # edge case of spawning in a terminal state
        if is_done(l):
          reward = self._reward(l, False)
          P[s][UP] = [(1.0, s, reward, True)]
          P[s][RIGHT] = [(1.0, s, reward, True)]
          P[s][DOWN] = [(1.0, s, reward, True)]
          P[s][LEFT] = [(1.0, s, reward, True)]
          P[s][STAY] = [(1.0, s, reward, True)]
        # not a terminal state
        else:
          # UP
          if y == 0:
            ns_up = s # hit wall -> stay
            nl_up = l
          else:
            nl_up = l - self.max_x
            if self._locked(nl_up, p):
              ns_up = s # door locked -> stay
              nl_up = l
            else:
              newp = self._update_key(p,nl_up)
              ns_up = self.loc_and_key_to_state[(nl_up,newp)]
          # RIGHT
          if x == (self.max_x - 1):
            ns_right = s # hit wall -> stay
            nl_right = l
          else:
            nl_right = l + 1
            if self._locked(nl_right, p):
              ns_right = s # door locked -> stay
              nl_right = l
            else:
              newp = self._update_key(p,nl_right)
              ns_right = self.loc_and_key_to_state[(nl_right,newp)]
          # DOWN
          if y == (self.max_y - 1):
            ns_down = s # hit wall -> stay
            nl_down = l
          else:
            nl_down = l + self.max_x
            if self._locked(nl_down, p):
              ns_down = s # door locked -> stay
              nl_down = l
            else:
              newp = self._update_key(p,nl_down)
              ns_down = self.loc_and_key_to_state[(nl_down,newp)]
          # LEFT
          if x == 0:
            ns_left = s # hit wall -> stay
            nl_left = l
          else:
            nl_left = l - 1
            if self._locked(nl_left, p):
              ns_left = s # door locked -> stay
              nl_left = l
            else:
              newp = self._update_key(p,nl_left)
              ns_left = self.loc_and_key_to_state[(nl_left,newp)]
            
          # random transitions
          if self.p_rand > 0:
            rand_trans = [(self.p_rand/self.nA, ns_up, self._reward(ns_up, False), is_done(ns_up)),
                          (self.p_rand/self.nA, ns_right, self._reward(ns_right, False), is_done(ns_right)),
                          (self.p_rand/self.nA, ns_down, self._reward(ns_down, False), is_done(ns_down)),
                          (self.p_rand/self.nA, ns_left, self._reward(ns_left, False), is_done(ns_left)),
                          (self.p_rand/self.nA, s, self._reward(s, False), is_done(s))]
          else: rand_trans = []
          
          # action-dependent transition probabilities, including random transitions
          P[s][UP] = [(1.0 - self.p_rand, ns_up, self._reward(nl_up, ns_up==s), is_done(nl_up))] + rand_trans
          P[s][RIGHT] = [(1.0 - self.p_rand, ns_right, self._reward(nl_right, ns_right==s), is_done(nl_right))] + rand_trans
          P[s][DOWN] = [(1.0 - self.p_rand, ns_down, self._reward(nl_down, ns_down==s), is_done(nl_down))] + rand_trans
          P[s][LEFT] = [(1.0 - self.p_rand, ns_left, self._reward(nl_left, ns_left==s), is_done(nl_left))] + rand_trans
          P[s][STAY] = [(1.0 - self.p_rand, s, self._reward(l, False), is_done(l))] + rand_trans
  
      it.iternext()
      
    self.P = P
      
    # sample starting state
    l = np.random.choice(self.spawn_locs, 1, p = self.spawn_dist)[0]
    self.s = self.loc_and_key_to_state[(l,None)]

    self.lastaction = None
    return self.s, self.g
  
  def set_goal(self, goal):
    return self._reset(goal)

  def _render(self, mode = 'human', close = False, bob_state = None):
      if close: return

      outfile = StringIO() if mode == 'ansi' else sys.stdout

      grid = np.arange(self.nL).reshape(self.shape)
      it = np.nditer(grid, flags=['multi_index'])
      while not it.finished:
        l = it.iterindex
        y, x = it.multi_index

        if self.state_to_loc[self.s] == l:
          output = " A "
        elif l == self.goal_locs[self.g]:
          output = " + "
        elif l in self.goal_locs:
          output = " - "
        elif l in self.loc_to_key:
          if self.loc_to_key[l] == 'master':
            output = " U "
          elif self.loc_to_key[l] == self.g:
            output = " P "
          else:
            output = " M "
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
  
#  import getch
  import time 
  button_delay = 0.2
   
  env = KeyGame(shape = [5,4],
                r_correct = +1,
                r_incorrect = -1,
                r_step = 0.,
                r_wall = -.1,
                p_rand = 0,
                spawn_locs = [9],
                spawn_dist = None,
                goal_locs = None,
                goal_dist = None,
                key_locs = [[1],[17]],
                master_key_locs = None)
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
    p = env.state_to_key[s]
    print('action: {}, reward: {}, key state: {}'.format(index_to_action[a], r, p))
    env._render()
    time.sleep(button_delay)