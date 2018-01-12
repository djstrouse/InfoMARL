# InfoMARL

Here we use information regularization to promote cooperation / competition in a multi-agent RL problem. The environment is a simple, two-goal grid world built in [OpenAI Gym](https://github.com/openai/gym) based on the example [here](https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/gridworld.py). The first agent has access to the goal, is parameterized with a tabular policy and value function, and is trained using REINFORCE, based on an implementation [here](https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/CliffWalk%20REINFORCE%20with%20Baseline%20Solution.ipynb). Its policy is regularized with the mutual information between goal and action, I(goal; action). The second agent does not have access to the goal, but instead must infer it from the behavior of the first agent.

TODOS:
*	save figures to results directory
*	basic figures for agent 1
*	implement agent 2