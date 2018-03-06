
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
#  alice_experiments = ['2018_02_07_1751_alice_positive_beta_cooperative_3x3',
#                       '2018_02_07_1807_alice_zero_beta_ambivalent_3x3',
#                       '2018_02_07_1823_alice_negative_beta_competitive_3x3',
#                       '2018_02_07_1839_alice_positive_beta_cooperative_3x3',
#                       '2018_02_07_1855_alice_zero_beta_ambivalent_3x3',
#                       '2018_02_07_2112_alice_negative_beta_competitive_3x3',
#                       '2018_02_07_2128_alice_positive_beta_cooperative_3x3',
#                       '2018_02_07_2144_alice_zero_beta_ambivalent_3x3',
#                       '2018_02_07_2200_alice_negative_beta_competitive_3x3',
#                       '2018_02_07_2216_alice_positive_beta_cooperative_3x3',
#                       '2018_02_07_2232_alice_zero_beta_ambivalent_3x3',
#                       '2018_02_07_2248_alice_negative_beta_competitive_3x3',
#                       '2018_02_07_2304_alice_positive_beta_cooperative_3x3',
#                       '2018_02_07_2331_alice_zero_beta_ambivalent_3x3',
#                       '2018_02_07_2348_alice_negative_beta_competitive_3x3']
  alice_experiments = ['job16290271_task0_2018_03_01_0200_alice_positive_beta_cooperative_5x5',
                       'job16290271_task1_2018_03_01_0159_alice_zero_beta_ambivalent_5x5',
                       'job16290271_task2_2018_03_01_0200_alice_negative_beta_competitive_5x5',
                       'job16290271_task3_2018_03_01_0200_alice_positive_beta_cooperative_5x5',
                       'job16290271_task4_2018_03_01_0200_alice_zero_beta_ambivalent_5x5',
                       'job16290271_task5_2018_03_01_0200_alice_negative_beta_competitive_5x5',
                       'job16290271_task6_2018_03_01_0201_alice_positive_beta_cooperative_5x5',
                       'job16290271_task7_2018_03_01_0201_alice_zero_beta_ambivalent_5x5',
                       'job16290271_task8_2018_03_01_0201_alice_negative_beta_competitive_5x5',
                       'job16290271_task9_2018_03_01_0201_alice_positive_beta_cooperative_5x5',
                       'job16290271_task10_2018_03_01_0202_alice_zero_beta_ambivalent_5x5',
                       'job16290271_task11_2018_03_01_0202_alice_negative_beta_competitive_5x5',
                       'job16290271_task12_2018_03_01_0202_alice_positive_beta_cooperative_5x5',
                       'job16290271_task13_2018_03_01_0202_alice_zero_beta_ambivalent_5x5',
                       'job16290271_task14_2018_03_01_0203_alice_negative_beta_competitive_5x5']
  bob_experiments = ['bob_with_cooperative_alice_shared128_200k',
                     'bob_with_ambivalent_alice_shared128_200k',
                     'bob_with_competitive_alice_shared128_200k']
  for i in range(15):
    replacements = {3: "experiment_name = '{}'".format(bob_experiments[i%3]),
                    5: "alice_experiment = '{}'".format(alice_experiments[i])}
    gen_bob_config(replacements = replacements, config_ext = str(i))