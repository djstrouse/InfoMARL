# InfoMARL

Here we use information regularization to promote cooperation / competition via intention signalling / hiding in a multi-agent RL problem. The environment is a simple, two-goal grid world built in [OpenAI Gym](https://github.com/openai/gym) based on the example [here](https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/gridworld.py). The first agent, Alice, has access to the goal, is parameterized with a tabular policy and value function, and is trained using REINFORCE, based on an implementation [here](https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/CliffWalk%20REINFORCE%20with%20Baseline%20Solution.ipynb). Alice's policy is regularized with the mutual information between goal and action (given state), I(goal; action | state). Depending on the sign of the information weighting, this regularization encourages her to either signal or hide her private information about the goal. The second agent, Bob, does not have access to the goal, but instead must infer it purely from observing the behavior of Alice. Thus, information regularization of Alice directly affects the success of Bob. In summary, information regularization allows Alice to train alone, but to be prepared for cooperation / competition with a friend / foe (Bob) introduced later. More detailed notes can be found [here](http://djstrouse.com/downloads/infomarl.pdf).

TODOS:
*	make richer episode visualization

OPTIONAL:
*	try discounting kl / entropy into future (like Distral paper); for high enough beta, Alice should try not to terminate episodes
*	under what conditions might Alice "overshoot" to signal?
*	under what conditions are I(traj;goal) and I(action;goal|state) approx equal?