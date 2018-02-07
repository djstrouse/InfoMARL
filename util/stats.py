import numpy as np

def rate_last_N(x, y, N = None):
  """Calculates the rate of increase in y per increase in x over the final
  stretch covering an increase in N of x. Assumes x monotonically increasing.
  if N is None, calculates rate over the entirety of x."""
  
  if N is None: index = 0
  else: index = np.argmax(x>(x[-1]-N))
  delta_x = x[-1]-x[index]
  delta_y = y[-1]-y[index]
  
  return delta_y/delta_x
  