
def gen_config(conf, replacements, config_ext):
  """Automatically generates new bob config files.
  
  Args:
    agent = alice or bob or env
    replacements = dict mapping line # to string to write there
    config_ext = extension to add to file name bob_config"""
  
  # read in base file to be modified
  with open(conf+'_config.py', 'r') as file:
    # read a list of lines into data
    lines = file.readlines()
    
  # now change the 2nd line, note that you have to add a newline
  for k in replacements.keys():
    lines[k] = replacements[k]+'\n'

  # and write everything back
  with open(conf+'_config'+config_ext+'.py', 'w') as file:
    file.writelines(lines)
      
  return
      
if __name__ == "__main__":
  conf = 'env'
  if conf == 'bob':
#    alice_experiments = ['job16290271_task0_2018_03_01_0200_alice_positive_beta_cooperative_5x5',
#                         'job16290271_task1_2018_03_01_0159_alice_zero_beta_ambivalent_5x5',
#                         'job16290271_task2_2018_03_01_0200_alice_negative_beta_competitive_5x5',
#                         'job16290271_task3_2018_03_01_0200_alice_positive_beta_cooperative_5x5',
#                         'job16290271_task4_2018_03_01_0200_alice_zero_beta_ambivalent_5x5',
#                         'job16290271_task5_2018_03_01_0200_alice_negative_beta_competitive_5x5',
#                         'job16290271_task6_2018_03_01_0201_alice_positive_beta_cooperative_5x5',
#                         'job16290271_task7_2018_03_01_0201_alice_zero_beta_ambivalent_5x5',
#                         'job16290271_task8_2018_03_01_0201_alice_negative_beta_competitive_5x5',
#                         'job16290271_task9_2018_03_01_0201_alice_positive_beta_cooperative_5x5',
#                         'job16290271_task10_2018_03_01_0202_alice_zero_beta_ambivalent_5x5',
#                         'job16290271_task11_2018_03_01_0202_alice_negative_beta_competitive_5x5',
#                         'job16290271_task12_2018_03_01_0202_alice_positive_beta_cooperative_5x5',
#                         'job16290271_task13_2018_03_01_0202_alice_zero_beta_ambivalent_5x5',
#                         'job16290271_task14_2018_03_01_0203_alice_negative_beta_competitive_5x5']
#    bob_experiments = ['bob_with_cooperative_alice_shared128_200k',
#                       'bob_with_ambivalent_alice_shared128_200k',
#                       'bob_with_competitive_alice_shared128_200k']
#    for i in range(15):
#      replacements = {3: "experiment_name = '{}'".format(bob_experiments[i%3]),
#                      5: "alice_experiment = '{}'".format(alice_experiments[i])}
#      gen_config(agent = agent, replacements = replacements, config_ext = str(i))
#    alice_experiments = ['job17338261_task90_2018_04_30_1617_alice_negative_state_competitive_1M_5x5',
#                         'job17338261_task91_2018_04_30_1645_alice_negative_state_competitive_1M_5x5',
#                         'job17338261_task92_2018_04_30_1643_alice_negative_state_competitive_1M_5x5',
#                         'job17338261_task93_2018_04_30_1618_alice_negative_state_competitive_1M_5x5',
#                         'job17338261_task94_2018_04_30_1618_alice_negative_state_competitive_1M_5x5',
#                         'job17338261_task100_2018_04_30_1618_alice_positive_state_cooperatitive_1M_5x5',
#                         'job17338261_task101_2018_04_30_1618_alice_positive_state_cooperatitive_1M_5x5',
#                         'job17338261_task103_2018_04_30_1736_alice_positive_state_cooperatitive_1M_5x5',
#                         'job17338261_task105_2018_04_30_1643_alice_positive_state_cooperatitive_1M_5x5',
#                         'job17338261_task106_2018_04_30_1644_alice_positive_state_cooperatitive_1M_5x5',
#                         'job17338261_task111_2018_04_30_1618_alice_unregularized_state_ambivalent_1M_5x5',
#                         'job17338261_task112_2018_04_30_1617_alice_unregularized_state_ambivalent_1M_5x5',
#                         'job17338261_task113_2018_04_30_1617_alice_unregularized_state_ambivalent_1M_5x5',
#                         'job17338261_task114_2018_04_30_1618_alice_unregularized_state_ambivalent_1M_5x5',
#                         'job17338261_task115_2018_04_30_1708_alice_unregularized_state_ambivalent_1M_5x5']
    alice_experiments = ['job17338261_task60_2018_04_30_1604_alice_negative_state_competitive_500k_5x5',
                         'job17338261_task61_2018_04_30_1604_alice_negative_state_competitive_500k_5x5',
                         'job17338261_task62_2018_04_30_1605_alice_negative_state_competitive_500k_5x5',
                         'job17338261_task63_2018_04_30_1604_alice_negative_state_competitive_500k_5x5',
                         'job17338261_task64_2018_04_30_1604_alice_negative_state_competitive_500k_5x5',
                         'job17338261_task70_2018_04_30_1604_alice_positive_state_cooperatitive_500k_5x5',
                         'job17338261_task71_2018_04_30_1604_alice_positive_state_cooperatitive_500k_5x5',
                         'job17338261_task72_2018_04_30_1605_alice_positive_state_cooperatitive_500k_5x5',
                         'job17338261_task73_2018_04_30_1604_alice_positive_state_cooperatitive_500k_5x5',
                         'job17338261_task74_2018_04_30_1604_alice_positive_state_cooperatitive_500k_5x5',
                         'job17338261_task80_2018_04_30_1604_alice_unregularized_state_ambivalent_500k_5x5',
                         'job17338261_task81_2018_04_30_1604_alice_unregularized_state_ambivalent_500k_5x5',
                         'job17338261_task82_2018_04_30_1605_alice_unregularized_state_ambivalent_500k_5x5',
                         'job17338261_task83_2018_04_30_1605_alice_unregularized_state_ambivalent_500k_5x5',
                         'job17338261_task84_2018_04_30_1604_alice_unregularized_state_ambivalent_500k_5x5']
    bob_experiments = ['bob_with_competitive_state_alice_shared128_200k',
                       'bob_with_cooperative_state_alice_shared128_200k',
                       'bob_with_ambivalent_state_alice_shared128_200k']
    i = 0
    for b in range(3):
      for j in range(5):
        replacements = {3: "experiment_name = '{}'".format(bob_experiments[b]),
                        5: "alice_experiment = '{}'".format(alice_experiments[i])}
        gen_config(conf = conf, replacements = replacements, config_ext = str(i))
        i += 1
    
  elif conf == 'alice':
    reg_strengths = [.025, .05, .1, .2, .4]
    i = 0
    for r in reg_strengths:
      replacements = {3: "experiment_name = 'alice_positive_state_cooperatitive_{}'".format(r),
                      32: "state_info_reg_strength = {}".format(r)}
      gen_config(conf = conf, replacements = replacements, config_ext = str(i))
      i += 1
  elif conf == 'env':
    i = 0
    p_rands = [0,2,4,8,16,32]
    for p in p_rands:
      replacements = {2: "experiment_name_ext = '_8x4_prand%i'" % p,
                      19: "                     p_rand = %.2f," % (p/100)}
      gen_config(conf = conf, replacements = replacements, config_ext = str(i))
      i += 1
    