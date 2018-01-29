# InfoMARL

Here we use information regularization to promote cooperation / competition via intention signalling / hiding in a multi-agent RL problem. The environment is a simple, two-goal grid world built in [OpenAI Gym](https://github.com/openai/gym) based on the example [here](https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/gridworld.py). The first agent has access to the goal, is parameterized with a tabular policy and value function, and is trained using REINFORCE, based on an implementation [here](https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/CliffWalk%20REINFORCE%20with%20Baseline%20Solution.ipynb). Its policy is regularized with the mutual information between goal and action (given state), I(goal; action | state). Depending on the sign of the information weighting, this regularization encourages the first agent to either signal or hide its private information about the goal. The second agent does not have access to the goal, but instead must infer it purely from the behavior of the first agent. Thus, information regularization of the first agent directly affects the success of the second agent. In summary, information regularization allows the first agent to train alone, but to be prepared for cooperation / competition with a friend / foe introduced later. More detailed notes can be found [here](http://djstrouse.com/downloads/infomarl.pdf).

TODOS:
*	implement training for agent 2

OPTIONAL:
*	try discounting kl / entropy into future (like Distral paper); for high enough beta, agent should try not to terminate episodes
*	under what conditions might agent 1 "overshoot" to signal?
*	under what conditions are I(traj;goal) and I(action;goal|state) approx equal?