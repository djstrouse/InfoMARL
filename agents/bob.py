import math
import numpy as np
import tensorflow as tf
ds = tf.contrib.distributions

class RNNObserver():
    """Processes observations of another agent with RNN to create 'belief state'
    over goal. Own state plus RNN output is fed into two networks to produce
    policy and value function. Trained with REINFORCE."""
    
    def __init__(self, env, policy_layer_sizes, value_layer_sizes, learning_rate = 0.025):
        
        with tf.variable_scope('observer_agent'):                       
            self.obs_states = tf.placeholder(tf.int32, [None], name = "observed_states")
            self.obs_actions = tf.placeholder(tf.int32, [None], name = "observed_actions")
            self.state = tf.placeholder(tf.int32, [], name = 'self_state')
            self.action = tf.placeholder(tf.int32, [], name = 'self_action')
            self.target = tf.placeholder(tf.float32, [], name = "target")           
            self.entropy_scale = tf.placeholder(tf.float32, [], name = "entropy_scale")
            self.value_scale = tf.placeholder(tf.float32, [], name = "value_scale")

            # rnn processing observations of other agent behaving
            with tf.variable_scope('rnn'):
                rnn_inputs = tf.one_hot(self.obs_states * env.nA + self.obs_actions,
                                        depth = env.nA * env.nS) # one-hot of (s,a)
                cell = tf.contrib.rnn.BasicRNNCell(1) # scalar core state
                # rnn_inputs must be 1 X t x d, where d = nS*nA
                _, z = tf.nn.dynamic_rnn(cell,
                                         tf.expand_dims(rnn_inputs,0),
                                         dtype = tf.float32)
                self.z = tf.squeeze(z, axis = 0) # drop batch_size=1 dim
                
            # concat agent state and rnn output as input to policy/value heads
            one_hot_state = tf.one_hot(self.state, depth = env.nS)
            x = tf.expand_dims(tf.concat([one_hot_state, self.z], axis = 0), 0)
                
            # policy head (fully-connected MLP)
            with tf.variable_scope('policy'):
                x_policy = x
                for n in policy_layer_sizes:
                    x_policy = tf.layers.dense(x_policy, n, activation = tf.nn.relu)
                logits = tf.layers.dense(x_policy, env.nA, activation = tf.nn.relu)
                self.action_probs = tf.squeeze(tf.nn.softmax(logits), axis = 0)
                self.picked_action_prob = tf.gather(self.action_probs, self.action)
                self.action_entropy = ds.Categorical(probs = self.action_probs).entropy()
                
            # value head (fully-connected MLP)
            with tf.variable_scope('value'):
                x_value = x
                for n in value_layer_sizes:
                    x_value = tf.layers.dense(x_value, n, activation = tf.nn.relu)
                self.value = tf.squeeze(tf.layers.dense(x_value, 1, activation = None))
            
            # loss and train op
            self.loss = -tf.log(self.picked_action_prob) * self.target + \
                        -self.entropy_scale * self.action_entropy + \
                        self.value_scale * tf.squared_difference(self.value, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step = tf.contrib.framework.get_global_step())
            
    def get_z(self, obs_states, obs_actions, state, sess = None):
        sess = sess or tf.get_default_session()
        return sess.run(self.z, {self.obs_states: obs_states,
                                 self.obs_actions: obs_actions,
                                 self.state: state})
    
    def predict(self, obs_states, obs_actions, state, sess = None):
        sess = sess or tf.get_default_session()
        return sess.run([self.action_probs, self.value],
                        {self.obs_states: obs_states,
                         self.obs_actions: obs_actions,
                         self.state: state})

    def update(self, obs_states, obs_actions, state, action, target,
               entropy_scale, value_scale, sess = None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.obs_states: obs_states,
                     self.obs_actions: obs_actions,
                     self.state: state,
                     self.action: action,
                     self.target: target,
                     self.entropy_scale: entropy_scale,
                     self.value_scale: value_scale}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss