from collections import namedtuple

experiment_name_ext = '_5x5'
env_type = 'grid'
agent = 'alice'

if env_type == 'grid':
  # TwoGoalGridWorld environment variables
  EnvParam = namedtuple('EnvironmentParameters',
                       ['shape',
                        'r_correct',
                        'r_incorrect',
                        'r_step',
                        'r_wall',
                        'p_rand',
                        'goal_locs',
                        'goal_dist'])
  env_param = EnvParam(shape = [5,5],
                       r_correct = +1,
                       r_incorrect = -1,
                       r_step = 0.,
                       r_wall = -.1,
                       p_rand = 0,
                       goal_locs = None,
                       goal_dist = None)
elif env_type == 'key':
  # KeyGame environment variables
  EnvParam = namedtuple('EnvironmentParameters',
                       ['shape',
                        'r_correct',
                        'r_incorrect',
                        'r_step',
                        'r_wall',
                        'p_rand',
                        'spawn_locs',
                        'spawn_dist',
                        'goal_locs',
                        'goal_dist',
                        'key_locs',
                        'master_key_locs'])
  if agent == 'alice':
    env_param = EnvParam(shape = [5,4],
                         r_correct = +1,
                         r_incorrect = -1,
                         r_step = 0.,
                         r_wall = -.1,
                         p_rand = 0,
                         spawn_locs = [6,10,14],
                         spawn_dist = None,
                         goal_locs = None,
                         goal_dist = None,
                         key_locs = [[2],[18]],
                         master_key_locs = [11])
  elif agent == 'bob':
    env_param = EnvParam(shape = [5,4],
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

def get_config():
    return env_type, env_param, experiment_name_ext