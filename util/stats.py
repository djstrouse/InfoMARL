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

def mean_last_N(x, y, N = None):
  """Calculates the mean of y for the last N steps/time measured by x. Assumes
  x monotonically increasing. If N is None, calculates mean of all of y."""
  
  if N is None: index = 0
  else: index = np.argmax(x>(x[-1]-N))
  
  return np.mean(y[index:])

def first_time_to(x, y):
  """x is considered time or steps, y reward. This function returns the total
  number of steps measured by x until reached 1,2,3,...,max(y) reward."""
  
  cum_y = np.cumsum(y)
  cum_x = np.cumsum(x)
  
  # calculate total time passed to reach level n
  total_time = []
  # loop over reward levels 1,2,3,...,max(y)
  for n in range(int(max(cum_y))):
    index = np.argmax(cum_y>n) # find first index of y that exceeds n
    total_time.append(cum_x[index]) # append cumulative time to reach n
    cum_y = cum_y[index+1:] # slice off searched part of cum_y
    cum_x = cum_x[index+1:] # slice off corresponding part of cum_x
  
  # calculate time passed between reaching levels n-1 and n
  total_time = np.asarray(total_time)
  time_between = total_time[1:]-total_time[:-1]
  time_between = np.insert(time_between,0,total_time[0])
    
  # typically, you'll want to plot total_time on x-axis, time_between on y-axis
  return total_time, time_between