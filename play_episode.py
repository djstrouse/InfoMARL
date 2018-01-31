import itertools
import copy
import numpy as np

def play(env, alice, bob, max_episode_length = 100):
  
  alice_env = env
  bob_env = copy.copy(alice_env)
  
  alice_state, goal = alice_env.reset()
  bob_state, _ = bob_env.set_goal(goal)
  
  alice_states = []
  alice_actions = []
  alice_done = False
  alice_total_reward = 0
  alice_episode_length = 0
  alice_total_kl = 0
  draw_alice = True
  
  bob_episode = []
  bob_done = False
  bob_total_reward = 0
  bob_episode_length = 0
  draw_bob = True
  
  # draw initial env
  alice_env._render(bob_state = bob_state)
  print('')
  
  # one step in the environment
  for t in itertools.count(start = 1):
    
    # first alice takes a step
    if not alice_done:
      alice_action_probs = alice.predict(alice_state, goal)
      alice_action = np.random.choice(np.arange(len(alice_action_probs)), p = alice_action_probs)
      kl = alice.get_kl(alice_state, goal)
      next_alice_state, alice_reward, alice_done, _ = alice_env.step(alice_action)
      # update alice stats
      alice_total_reward += alice_reward
      alice_episode_length = t
      alice_total_kl += kl
    else: # if done, sit still
      alice_action = alice_env.action_to_index['STAY']
      next_alice_state = alice_state
    alice_states.append(alice_state)
    alice_actions.append(alice_action)
    
    # draw env with alice step
    if draw_alice:
      alice_env._render(bob_state = bob_state)
      print('alice step %i: reward = %i, total kl = %.2f' % (t, alice_total_reward, alice_total_kl))
      print('')
      if alice_done: draw_alice = False # only draw alice step first step after done
    
    
    # then bob takes a step
    if not bob_done:
      bob_action_probs, bob_value = bob.predict(alice_states, alice_actions, bob_state)
      bob_action = np.random.choice(np.arange(len(bob_action_probs)), p = bob_action_probs)
      z = bob.get_z(alice_states, alice_actions, bob_state)
      next_bob_state, bob_reward, bob_done, _ = bob_env.step(bob_action)
      bob_total_reward += bob_reward
      bob_episode_length = t
    else: # if done, sit still
      bob_next_state = bob_state
    
    # draw env with bob step
    if draw_bob:
      alice_env._render(bob_state = next_bob_state)
      print('bob step %i: reward = %i, rnn latent = %.2f' % (t, bob_total_reward, z))
      print('')
      if bob_done: draw_bob = False # only draw bob step first step after done

    if (alice_done and bob_done) or t > max_episode_length: break
        
    alice_state = next_alice_state
    bob_state = next_bob_state
  