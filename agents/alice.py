import math
import numpy as np
import tensorflow as tf
ds = tf.contrib.distributions

class PolicyEstimator():
    """Tabular multi-goal policy with entropy reguarlization and information
    regularization trained by REINFORCE."""
    
    def __init__(self, env, learning_rate = 0.025, scope = "policy"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.int32, [], name = "state")
            self.goal = tf.placeholder(tf.int32, [], name = "goal")
            self.action = tf.placeholder(dtype = tf.int32, name = "action")
            self.target = tf.placeholder(dtype = tf.float32, name = "target")
            self.entropy_scale = tf.placeholder(dtype = tf.float32, name = "entropy_scale")
            self.beta = tf.placeholder(dtype = tf.float32, name = 'beta')

            # tabular mapping from (state,goal) to action
            self._logits = tf.Variable(tf.random_normal([env.nG, env.nS, env.nA], stddev = .1), name='policy_logits')
            self.action_probs = tf.nn.softmax(tf.squeeze(self._logits[self.goal, self.state]))
            
            # additional useful quantities
            self.base_action_probs = tf.reduce_mean(tf.nn.softmax(tf.squeeze(self._logits[:, self.state])), axis = 0) # ASSUMES UNIFORM P(G)
            self.kl = ds.kl_divergence(ds.Categorical(probs = self.action_probs),
                                       ds.Categorical(probs = self.base_action_probs))/math.log(2) # in bits          
            self.action_entropy = ds.Categorical(probs = self.action_probs).entropy()
            self.picked_action_prob = tf.gather(self.action_probs, self.action)

            # Loss and train op
            self.loss = -tf.log(self.picked_action_prob) * (self.target) + \
                        -self.entropy_scale * self.action_entropy + \
                        -self.beta * self.kl

            self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step = tf.contrib.framework.get_global_step())
            
    def get_kl(self, state, goal, sess = None):
        sess = sess or tf.get_default_session()
        return sess.run(self.kl, {self.state: state, self.goal: goal})
    
    def predict(self, state, goal, sess = None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_probs, {self.state: state, self.goal: goal})

    def update(self, state, goal, target, action, entropy_scale, beta, sess = None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state,
                     self.goal: goal,
                     self.target: target,
                     self.action: action,
                     self.entropy_scale: entropy_scale,
                     self.beta: beta}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


class ValueEstimator():
    """Tabular value function for a multi-goal environment."""
    
    def __init__(self, env, learning_rate = 0.05, scope = "value"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.int32, [], "state")
            self.goal = tf.placeholder(tf.int32, [], name = "goal")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            # tabular mapping from (goal, state) to value
            self.value_estimates = tf.Variable(tf.random_normal([env.nG, env.nS], stddev = .1), name='value_estimates')
            
            self.value = tf.squeeze(self.value_estimates[self.goal, self.state])
            self.loss = tf.squared_difference(self.value, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())        
    
    def predict(self, state, goal, sess = None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value, {self.state: state, self.goal: goal})

    def update(self, state, goal, target, sess = None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state, self.goal: goal, self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss
    
def get_action_probs(policy_estimator, env, sess):
  """"Extracts policy array from agent."""
  action_probs = np.ones((env.nS, env.nG, env.nA))
  #base_action_probs = np.ones((env.nS, env.nA))
  for s in range(env.nS):
    for g in range(env.nG):
      action_probs[s,g,:] = policy_estimator.predict(s, g, sess)
    #base_action_probs[s,:]
  return action_probs

def get_values(value_estimator, env, sess):
  """"Extracts value array from agent."""
  values = np.ones((env.nS, env.nG))
  for g in range(env.nG):
    for s in range(env.nS):
      # correct/incorrect goal marked by negative numbers for plotting purposes
      if s == env.goal_locs[g]: val = -.5
      elif s in env.goal_locs: val = -1
      # for other states, use value estimate
      else: val = value_estimator.predict(s, g, sess)
      values[s,g] = val
  return values

def get_kls(policy_estimator, env, sess):
  """"Extracts info array from agent."""
  # since policy not updated in terminal states, set those to rewards
  kls = np.ones((env.nS, env.nG))
  for g in range(env.nG):
    for s in range(env.nS):
      # correct/incorrect goal marked by negative numbers for plotting purposes
      if s == env.goal_locs[g]: kl = -.5
      elif s in env.goal_locs: kl = -1
      # for other states, use info estimate
      else: kl = policy_estimator.get_kl(s, g, sess)
      kls[s,g] = kl
  return kls