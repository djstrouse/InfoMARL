import math
import time
import numpy as np
import tensorflow as tf
ds = tf.contrib.distributions
from collections import namedtuple

Losses = namedtuple('Losses', ['loss', 'pg_loss', 'ent_loss', 'value_loss',
                               'action_info_loss', 'state_info_loss'])
DebuggingStats = namedtuple('DebuggingStats', ['imp_samp_weights',
                                               'cf_state_probs',
                                               'this_state_prob',
                                               'cf_policy',
                                               'this_policy',
                                               'Rstate',
                                               'Rcf'])

class TabularREINFORCE:
    """Tabular multi-goal policy with entropy reguarlization and information
    regularization trained by REINFORCE."""
    
    def __init__(self, env, use_action_info = True, use_state_info = True, policy = None):
      
      self.use_action_info = use_action_info
      self.use_state_info = use_state_info
      
      self.state = tf.placeholder(tf.int32, [], name = 'state')
      self.goal = tf.placeholder(tf.int32, [], name = 'goal')
      
      if policy is not None:
        self.action_probs = policy
        self.trainable = False
      else:
        self.trainable = True
      
        self.action = tf.placeholder(tf.int32, name = 'action')
        self.return_estimate = tf.placeholder(tf.float32, name = 'return_estimate')
        self.learning_rate = tf.placeholder(tf.float32, [], name = 'learning_rate')
        self.entropy_scale = tf.placeholder(tf.float32, [], name = 'entropy_scale')
        self.value_scale = tf.placeholder(tf.float32, [], name = 'value_scale')
        self.action_info_scale = tf.placeholder(tf.float32, name = 'action_info_scale')
        self.state_info_scale = tf.placeholder(tf.float32, name = 'state_info_scale')
        self.state_goal_counts = tf.placeholder(tf.float32, [env.nS, env.nG], name = 'state_goal_counts')
        self.next_state = tf.placeholder(tf.int32, [], name = 'next_state')
  
        # policy: tabular mapping from (state,goal) to action
        self.raw_logits = tf.Variable(tf.random_normal([env.nG, env.nS, env.nA], stddev = .1), name = 'policy_logits')
        max_logits = tf.reduce_max(self.raw_logits, axis = 2, keep_dims = True)
        self.logits = self.raw_logits - max_logits
        self.these_logits = tf.squeeze(self.logits[self.goal, self.state])
        self.action_probs = tf.nn.softmax(self.these_logits)
        self.picked_action_prob = tf.gather(self.action_probs, self.action)
        
        # action info
        if use_action_info:
          self.base_action_probs = tf.reduce_mean(tf.nn.softmax(tf.squeeze(self.logits[:, self.state])), axis = 0) # ASSUMES UNIFORM P(G)
          self.kl = ds.kl_divergence(ds.Categorical(probs = self.action_probs),
                                     ds.Categorical(probs = self.base_action_probs))/math.log(2) # in bits          
          self.action_info_loss = -self.action_info_scale * self.kl
        else:
          self.action_info_loss = 0
        
        # state info (ASSUMES UNIFORM GOAL PROBABILITIES, hard-coded n=gamma=1)
        if use_state_info:
          this_state_count = self.state_goal_counts[self.state, self.goal]
          this_goal_count = tf.reduce_sum(self.state_goal_counts[:, self.goal])
          this_state_prob = this_state_count / this_goal_count # p(s_t-1|g)
          cf_state_counts = tf.squeeze(self.state_goal_counts[self.state, :]) # counterfactual counts
          cf_goal_counts = tf.reduce_sum(self.state_goal_counts, axis = 0) # counterfactual counts
          cf_state_probs = cf_state_counts / cf_goal_counts # p(s_t-1|g')
          self.cf_logits = tf.squeeze(self.logits[:, self.state, :])
          cf_policy = tf.squeeze(tf.nn.softmax(self.cf_logits)[:, self.action]) # counterfactual policy
          this_policy = self.picked_action_prob
          imp_samp_weights = (cf_state_probs / this_state_prob) * tf.stop_gradient(cf_policy / this_policy)
          next_state_count = self.state_goal_counts[self.next_state, self.goal]
          next_state_prob = next_state_count / this_goal_count # p(s_t|g)
          next_total_count = tf.reduce_sum(self.state_goal_counts[self.next_state, :])
          next_total_prob = next_total_count / tf.reduce_sum(self.state_goal_counts) # p(s_t)      
          next_state_ratio = next_state_prob / next_total_prob 
          Rstate = 1+tf.log(next_state_ratio) # doesn't include g' = g term
          Rcf = imp_samp_weights * next_state_ratio # includes g' = g term
          state_update = Rstate * tf.log(this_policy) / np.log(2)
          cf_update_terms = Rcf * tf.log(cf_policy)
          cf_update = tf.reduce_mean(cf_update_terms) / np.log(2)
          self.state_info_loss = -self.state_info_scale * (state_update - cf_update)
          self.dstats = DebuggingStats(imp_samp_weights = imp_samp_weights,
                                       cf_state_probs = cf_state_probs,
                                       this_state_prob = this_state_prob,
                                       cf_policy = cf_policy,
                                       this_policy = this_policy,
                                       Rstate = Rstate,
                                       Rcf = Rcf)
        else:
           self.state_info_loss = 0
           self.dstats = None
           
        # entropy bonus
        self.action_entropy = ds.Categorical(probs = self.action_probs).entropy()
        self.ent_loss = -self.entropy_scale * self.action_entropy
           
        # values: tabular mapping from (state,goal) to value
        self.value_estimates = tf.Variable(tf.random_normal([env.nG, env.nS], stddev = .1), name = 'value_estimates')
        self.value = tf.squeeze(self.value_estimates[self.goal, self.state])
        self.advantage = self.return_estimate - tf.stop_gradient(self.value)
        self.target = self.return_estimate
        if use_state_info: self.target += self.state_info_scale * (1 + tf.log(next_state_ratio))
        # AGAIN ASSUMES N=GAMMA=1, NEED TO ADD ADJUSTMENT FOR ACTION INFO ALSO
        self.value_loss = self.value_scale * tf.squared_difference(self.value, self.target)
        
        # REINFORCE loss
        self.pg_loss = -tf.log(self.picked_action_prob) * self.advantage
  
        # total loss and train op
        self.loss = self.pg_loss + self.action_info_loss + self.state_info_loss + \
                    self.ent_loss + self.value_loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op = self.optimizer.minimize(
            self.loss, global_step = tf.contrib.framework.get_global_step())
            
    def get_kl(self, state, goal, sess = None):
      sess = sess or tf.get_default_session()
      feed_dict = {self.state: state,
                   self.goal: goal}
      return sess.run(self.kl, feed_dict)
      
    def get_action_probs(self, state, goal, sess = None):
      if self.trainable:
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state,
                     self.goal: goal}
        return sess.run(self.action_probs, feed_dict)
      else:
        return self.action_probs[goal,state,:]
      
    def get_value(self, state, goal, sess = None):
      sess = sess or tf.get_default_session()
      feed_dict = {self.state: state,
                   self.goal: goal}
      return sess.run(self.value, feed_dict)
    
    def predict(self, state, goal, sess = None):
      if self.trainable:
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state,
                     self.goal: goal}
        return sess.run([self.action_probs, self.value, self.these_logits], feed_dict)
      else:
        return self.action_probs[goal,state,:], None

    def update(self, state, goal, action, return_estimate,
               learning_rate, entropy_scale, value_scale,
               action_info_scale = None, state_info_scale = None,
               state_goal_counts = None, next_state = None, sess = None):
      # action_info_scale required if use_action_info = True
      # state_info_scale, state_goal_counts, next_state req'd if use_state_info = True
      if self.trainable:
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state,
                     self.goal: goal,
                     self.action: action,
                     self.return_estimate: return_estimate,
                     self.learning_rate: learning_rate,
                     self.entropy_scale: entropy_scale,
                     self.value_scale: value_scale}
        if self.use_state_info:
          feed_dict[self.state_info_scale] = state_info_scale
          feed_dict[self.state_goal_counts] = state_goal_counts
          feed_dict[self.next_state] = next_state
        if self.use_action_info:
          feed_dict[self.action_info_scale] = action_info_scale    
#        feed_dict = {self.state: state,
#                     self.goal: goal,
#                     self.action: action,
#                     self.return_estimate: return_estimate,
#                     self.learning_rate: learning_rate,
#                     self.entropy_scale: entropy_scale,
#                     self.value_scale: value_scale,
#                     self.action_info_scale: action_info_scale,
#                     self.state_info_scale: state_info_scale,
#                     self.state_goal_counts: state_goal_counts,
#                     self.next_state: next_state}
        if self.use_state_info and self.use_action_info:
          _, loss, pg_loss, ai_loss, si_loss, ent_loss, v_loss, dstats = sess.run( \
               [self.train_op, self.loss, self.pg_loss, self.action_info_loss,
                self.state_info_loss, self.ent_loss, self.value_loss, self.dstats], feed_dict)
        elif self.use_state_info:
          # check logits of affected actions
          pre_logits = sess.run([self.cf_logits], feed_dict) # nG x nA
          # update
          _, loss, pg_loss, si_loss, ent_loss, v_loss, dstats = sess.run( \
               [self.train_op, self.loss, self.pg_loss,
                self.state_info_loss, self.ent_loss, self.value_loss, self.dstats], feed_dict)
          ai_loss = 0
          # check post-update logits
          post_logits = sess.run([self.cf_logits], feed_dict) # nG x nA
          if np.any(np.isnan(post_logits)):
            print('Logits transitioned to NaNs')
            print(pre_logits)
            print(post_logits)
            print(dstats)
            time.sleep(10)
          
        elif self.use_action_info:
          _, loss, pg_loss, ai_loss, ent_loss, v_loss = sess.run( \
               [self.train_op, self.loss, self.pg_loss, self.action_info_loss,
                self.ent_loss, self.value_loss], feed_dict)
          si_loss = 0
          dstats = None
        else:
          _, loss, pg_loss, ent_loss, v_loss = sess.run( \
               [self.train_op, self.loss, self.pg_loss, self.ent_loss,
                self.value_loss], feed_dict)
          ai_loss = 0
          si_loss = 0
        losses = Losses(loss = loss,
                        pg_loss = pg_loss,
                        ent_loss = ent_loss,
                        value_loss = v_loss,
                        action_info_loss = ai_loss,
                        state_info_loss = si_loss)
        return losses, dstats
      else:
        return None
    
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
  """Extracts action info array from agent."""
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