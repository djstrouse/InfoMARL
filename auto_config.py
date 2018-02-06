
def gen_bob_config(replacements, config_ext):
  """Automatically generates new bob config files.
  
  Args:
    replacements = dict mapping line # to string to write there
    config_ext = extension to add to file name bob_config"""
  
  # read in base file to be modified
  with open('bob_config.py', 'r') as file:
    # read a list of lines into data
    lines = file.readlines()
    
  # now change the 2nd line, note that you have to add a newline
  for k in replacements.keys():
    lines[k] = replacements[k]+'\n'

  # and write everything back
  with open('bob_config'+config_ext+'.py', 'w') as file:
    file.writelines(lines)
      
  return
      
if __name__ == "__main__":
  alice_experiments = ['2018_02_01_0151_small_positive_3x3',
                       '2018_01_31_2052_unregularized_3x3',
                       '2018_02_01_0400_small_negative_3x3',
                       '2018_02_01_0008_small_positive_3x3',
                       '2018_01_31_2307_unregularized_3x3',
                       '2018_02_01_0238_small_negative_3x3',
                       '2018_01_31_2339_small_positive_3x3',
                       '2018_01_31_2250_unregularized_3x3',
                       '2018_02_01_0416_small_negative_3x3']
  bob_experiments = ['bob_with_cooperative_alice_delayed_goal_64',
                     'bob_with_ambivalent_alice_delayed_goal_64',
                     'bob_with_competitive_alice_delayed_goal_64']
  for i in range(9):
    replacements = {3: "experiment_name = '{}'".format(bob_experiments[i%3]),
                    5: "alice_experiment = '{}'".format(alice_experiments[i])}
    gen_bob_config(replacements = replacements, config_ext = str(i))