import sys
if "../" not in sys.path:
  sys.path.append("../") 
from envs.TwoGoalGridWorld import TwoGoalGridWorld

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
STAY = 4
env = TwoGoalGridWorld(shape = [3,4],
                       r_correct = 1,
                       r_incorrect = -1,
                       r_step = 0,
                       r_wall = -.1,
                       p_rand = 0,
                       goal_locs = None,
                       goal_dist = None)

#print('move left into wall')
#print(env.P[4][LEFT])
#print(env.P[8][LEFT])
#print('move right from left wall')
#print(env.P[4][RIGHT])
#print(env.P[8][RIGHT])
#print('move right into wall')
#print(env.P[7][LEFT])
#print(env.P[11][LEFT])
#print('move left from right wall')
#print(env.P[7][RIGHT])
#print(env.P[11][RIGHT])
#print('move up into goal')
#print(env.P[4][UP])
#print(env.P[7][UP])

for i in range(12):
  print('state %i' % i)
  print(env.P[i])