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
import config

tf.reset_default_graph()

global_step = tf.Variable(0, name = "global_step", trainable = False)
env_param, agent_param, training_param, experiment_name = config.get_config()
env = TwoGoalGridWorld(env_param.shape,
                       env_param.r_correct,
                       env_param.r_incorrect,
                       env_param.r_step)
policy_estimator = PolicyEstimator(env)
value_estimator = ValueEstimator(env)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
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