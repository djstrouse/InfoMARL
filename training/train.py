import tensorflow as tf
import numpy as np
import sys
if "../" not in sys.path:
  sys.path.append("../") 
from envs.TwoGoalGridWorld import TwoGoalGridWorld
from agents.TabularREINFORCE import *
from training.REINFORCE import reinforce
from plotting.plot_episode_stats import *
from plotting.visualize_grid_world import *

tf.reset_default_graph()

global_step = tf.Variable(0, name = "global_step", trainable = False)
env = TwoGoalGridWorld(shape = [3,3])
policy_estimator = PolicyEstimator(env)
value_estimator = ValueEstimator(env)

# set parameters
num_episodes = 500
entropy_scale = np.logspace(np.log10(.2), np.log10(.01), num_episodes)
beta = [0]*num_episodes

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    stats = reinforce(env, policy_estimator, value_estimator,
                      num_episodes = num_episodes,
                      entropy_scale = entropy_scale,
                      beta = beta,
                      discount_factor = 0.8)
    values = get_values(value_estimator, env, sess) # state X goal
    kls = get_kls(policy_estimator, env, sess) # state X goal
    action_probs = get_action_probs(policy_estimator, env, sess) # state X goal X action
    
    
# plot results
plot_episode_stats(stats, smoothing_window = 25)
k = 15
print('-'*k+'VALUES'+'-'*k)
plot_value_map(values, action_probs, env)
print('-'*k+'KLS'+'-'*k)
plot_kl_map(kls, action_probs, env)
print('-'*k+'POLICY'+'-'*k)
print_policy(action_probs, env)