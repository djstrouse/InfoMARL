import math
import numpy as np
import tensorflow as tf
ds = tf.contrib.distributions

class TabularREINFORCE():
    """Tabular multi-goal policy with entropy reguarlization and information
    regularization trained by REINFORCE."""
    
    def __init__(self, env):
      
      self.state = tf.placeholder(tf.int32, [], name = "state")
      self.goal = tf.placeholder(tf.int32, [], name = "goal")
      self.action = tf.placeholder(tf.int32, name = "action")
      self.return_estimate = tf.placeholder(tf.float32, name = "return_estimate")
      self.learning_rate = tf.placeholder(tf.float32, [], name = "learning_rate")
      self.entropy_scale = tf.placeholder(tf.float32, [], name = "entropy_scale")
      self.value_scale = tf.placeholder(tf.float32, [], name = "value_scale")
      self.info_scale = tf.placeholder(tf.float32, name = 'beta')
      
      # values: tabular mapping from (state,goal) to value
      self.value_estimates = tf.Variable(tf.random_normal([env.nG, env.nS], stddev = .1), name = 'value_estimates')
      self.value = tf.squeeze(self.value_estimates[self.goal, self.state])
      self.advantage = self.return_estimate - tf.stop_gradient(self.value)
      self.value_loss = self.value_scale * tf.squared_difference(self.value, self.return_estimate)

      # policy: tabular mapping from (state,goal) to action
      self._logits = tf.Variable(tf.random_normal([env.nG, env.nS, env.nA], stddev = .1), name = 'policy_logits')
      self.action_probs = tf.nn.softmax(tf.squeeze(self._logits[self.goal, self.state]))
      self.base_action_probs = tf.reduce_mean(tf.nn.softmax(tf.squeeze(self._logits[:, self.state])), axis = 0) # ASSUMES UNIFORM P(G)
      self.kl = ds.kl_divergence(ds.Categorical(probs = self.action_probs),
                                 ds.Categorical(probs = self.base_action_probs))/math.log(2) # in bits          
      self.action_entropy = ds.Categorical(probs = self.action_probs).entropy()
      self.picked_action_prob = tf.gather(self.action_probs, self.action)
      self.pg_loss = -tf.log(self.picked_action_prob) * self.advantage
      self.info_loss = -self.info_scale * self.kl
      self.ent_loss = -self.entropy_scale * self.action_entropy

      # total loss and train op
      self.loss = self.pg_loss + self.info_loss + self.ent_loss + self.value_loss
      self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
      self.train_op = self.optimizer.minimize(
          self.loss, global_step = tf.contrib.framework.get_global_step())
            
    def get_kl(self, state, goal, sess = None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state,
                     self.goal: goal}
        return sess.run(self.kl, feed_dict)
      
    def get_action_probs(self, state, goal, sess = None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state,
                     self.goal: goal}
        return sess.run(self.action_probs, feed_dict)
      
    def get_value(self, state, goal, sess = None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state,
                     self.goal: goal}
        return sess.run(self.value, feed_dict)
    
    def predict(self, state, goal, sess = None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state,
                     self.goal: goal}
        return sess.run([self.action_probs, self.value, self.kl], feed_dict)

    def update(self, state, goal, action, return_estimate,
               learning_rate,  entropy_scale, value_scale, info_scale,
               sess = None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state,
                     self.goal: goal,
                     self.action: action,
                     self.return_estimate: return_estimate,
                     self.learning_rate: learning_rate,
                     self.entropy_scale: entropy_scale,
                     self.value_scale: value_scale,
                     self.info_scale: info_scale}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss
    
def get_action_probs(agent, env, sess):
  """"Extracts policy array from agent."""
  action_probs = np.ones((env.nS, env.nG, env.nA))
  #base_action_probs = np.ones((env.nS, env.nA))
  for s in range(env.nS):
    for g in range(env.nG):
       action_probs[s,g,:] = agent.get_action_probs(s, g, sess)
    #base_action_probs[s,:]
  return action_probs

def get_values(agent, env, sess):
  """"Extracts value array from agent."""
  values = np.ones((env.nS, env.nG))
  for g in range(env.nG):
    for s in range(env.nS):
      # correct/incorrect goal marked by negative numbers for plotting purposes
      if s == env.goal_locs[g]: val = -.5
      elif s in env.goal_locs: val = -1
      # for other states, use value estimate
      else: val = agent.get_value(s, g, sess)
      values[s,g] = val
  return values

def get_kls(agent, env, sess):
  """"Extracts info array from agent."""
  # since policy not updated in terminal states, set those to rewards
  kls = np.ones((env.nS, env.nG))
  for g in range(env.nG):
    for s in range(env.nS):
      # correct/incorrect goal marked by negative numbers for plotting purposes
      if s == env.goal_locs[g]: kl = -.5
      elif s in env.goal_locs: kl = -1
      # for other states, use info estimate
      else: kl = agent.get_kl(s, g, sess)
      kls[s,g] = kl
  return kls