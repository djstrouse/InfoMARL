def test_random_agent(env):
  state = env.reset()
  total_reward = 0
  env._render()
  # loop over steps
  for t in itertools.count():
    # choose random action
    action = np.random.choice(np.arange(env.nA))
    # feed to environment
    next_state, reward, done, _ = env.step(action)
    total_reward += reward
    # print summary and render
    print("\rStep {}: Reward = {:.1f}, Cumulative = {:.1f}, Action: {}".format(t, reward, total_reward, index_to_action[action]))
    env._render()
    # prepare for end of step
    if done: break
    state = next_state # not actually used for random agent