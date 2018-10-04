
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
  conf = 'alice'
  if conf == 'bob':
#    alice_experiments = ['job17566260_task0_2018_05_16_1433_alice_positive_action_cooperative_100k_5x5',
#                         'job17566260_task1_2018_05_16_1433_alice_positive_action_cooperative_100k_5x5',
#                         'job17566260_task2_2018_05_16_1433_alice_positive_action_cooperative_100k_5x5',
#                         'job17566260_task3_2018_05_16_1433_alice_positive_action_cooperative_100k_5x5',
#                         'job17566260_task4_2018_05_16_1434_alice_positive_action_cooperative_100k_5x5',
#                         'job17566260_task10_2018_05_16_1434_alice_negative_action_competitive_100k_5x5',
#                         'job17566260_task11_2018_05_16_1434_alice_negative_action_competitive_100k_5x5',
#                         'job17566260_task12_2018_05_16_1435_alice_negative_action_competitive_100k_5x5',
#                         'job17566260_task13_2018_05_16_1435_alice_negative_action_competitive_100k_5x5',
#                         'job17566260_task14_2018_05_16_1435_alice_negative_action_competitive_100k_5x5',
#                         'job17566260_task20_2018_05_16_1436_alice_unregularized_action_ambivalent_100k_5x5',
#                         'job17566260_task21_2018_05_16_1436_alice_unregularized_action_ambivalent_100k_5x5',
#                         'job17566260_task22_2018_05_16_1436_alice_unregularized_action_ambivalent_100k_5x5',
#                         'job17566260_task23_2018_05_16_1436_alice_unregularized_action_ambivalent_100k_5x5',
#                         'job17566260_task24_2018_05_16_1437_alice_unregularized_action_ambivalent_100k_5x5']
#    bob_experiments = ['bob_with_cooperative_action_alice_200k',
#                       'bob_with_competitive_action_alice_200k',
#                       'bob_with_ambivalent_action_alice_200k']
#    alice_experiments = ['job17553636_task30_2018_05_15_2216_alice_positive_state_cooperatitive_beta0.025_250k_5x5',
#                         'job17553636_task31_2018_05_15_2212_alice_positive_state_cooperatitive_beta0.025_250k_5x5',
#                         'job17553636_task32_2018_05_15_2213_alice_positive_state_cooperatitive_beta0.025_250k_5x5',
#                         'job17553636_task34_2018_05_15_2218_alice_positive_state_cooperatitive_beta0.025_250k_5x5',
#                         'job17553636_task35_2018_05_15_2216_alice_positive_state_cooperatitive_beta0.025_250k_5x5',
#                         'job17553636_task44_2018_05_15_2214_alice_negative_state_competitive_beta0.025_250k_5x5',
#                         'job17553636_task45_2018_05_15_2216_alice_negative_state_competitive_beta0.025_250k_5x5',
#                         'job17553636_task46_2018_05_15_2216_alice_negative_state_competitive_beta0.025_250k_5x5',
#                         'job17553636_task47_2018_05_15_2215_alice_negative_state_competitive_beta0.025_250k_5x5',
#                         'job17553636_task48_2018_05_15_2215_alice_negative_state_competitive_beta0.025_250k_5x5',
#                         'job17553636_task50_2018_05_15_2216_alice_unregularized_state_ambivalent_250k_5x5',
#                         'job17553636_task51_2018_05_15_2216_alice_unregularized_state_ambivalent_250k_5x5',
#                         'job17553636_task52_2018_05_15_2216_alice_unregularized_state_ambivalent_250k_5x5',
#                         'job17553636_task53_2018_05_15_2216_alice_unregularized_state_ambivalent_250k_5x5',
#                         'job17553636_task54_2018_05_15_2220_alice_unregularized_state_ambivalent_250k_5x5']
#    bob_experiments = ['bob_with_cooperative_state_alice_200k',
#                       'bob_with_competitive_state_alice_200k',
#                       'bob_with_ambivalent_state_alice_200k']
    
#    alice_experiments = ['job17583555_task101_2018_05_17_1630_alice_negative_action_competitive_beta0.2_discount0.8_250k_KeyGame',
#                         'job17583555_task105_2018_05_17_1632_alice_negative_action_competitive_beta0.2_discount0.8_250k_KeyGame',
#                         'job17583555_task106_2018_05_17_1631_alice_negative_action_competitive_beta0.2_discount0.8_250k_KeyGame',
#                         'job17583555_task107_2018_05_17_1631_alice_negative_action_competitive_beta0.2_discount0.8_250k_KeyGame',
#                         'job17583555_task108_2018_05_17_1633_alice_negative_action_competitive_beta0.2_discount0.8_250k_KeyGame',
#                         'job17583555_task340_2018_05_17_1715_alice_positive_action_cooperative_beta0.2_discount0.8_250k_KeyGame',
#                         'job17583555_task341_2018_05_17_1715_alice_positive_action_cooperative_beta0.2_discount0.8_250k_KeyGame',
#                         'job17583555_task343_2018_05_17_1718_alice_positive_action_cooperative_beta0.2_discount0.8_250k_KeyGame',
#                         'job17583555_task344_2018_05_17_1717_alice_positive_action_cooperative_beta0.2_discount0.8_250k_KeyGame',
#                         'job17583555_task345_2018_05_17_1718_alice_positive_action_cooperative_beta0.2_discount0.8_250k_KeyGame',
#                         'job17583555_task210_2018_05_17_1655_alice_unregularized_action_ambivalent_discount0.8_250k_KeyGame',
#                         'job17583555_task211_2018_05_17_1653_alice_unregularized_action_ambivalent_discount0.8_250k_KeyGame',
#                         'job17583555_task212_2018_05_17_1653_alice_unregularized_action_ambivalent_discount0.8_250k_KeyGame',
#                         'job17583555_task213_2018_05_17_1654_alice_unregularized_action_ambivalent_discount0.8_250k_KeyGame',
#                         'job17583555_task214_2018_05_17_1654_alice_unregularized_action_ambivalent_discount0.8_250k_KeyGame']
#    bob_experiments = ['bob_with_competitive_action_alice',
#                       'bob_with_cooperative_action_alice',
#                       'bob_with_ambivalent_action_alice']
    
    alice_experiments = ['job17583555_task112_2018_05_17_1634_alice_negative_action_competitive_beta0.25_discount0.8_250k_KeyGame',
                         'job17583555_task114_2018_05_17_1632_alice_negative_action_competitive_beta0.25_discount0.8_250k_KeyGame',
                         'job17583555_task116_2018_05_17_1631_alice_negative_action_competitive_beta0.25_discount0.8_250k_KeyGame',
                         'job17583555_task117_2018_05_17_1633_alice_negative_action_competitive_beta0.25_discount0.8_250k_KeyGame',
                         'job17583555_task118_2018_05_17_1632_alice_negative_action_competitive_beta0.25_discount0.8_250k_KeyGame',
                         'job17583555_task351_2018_05_17_1718_alice_positive_action_cooperative_beta0.25_discount0.8_250k_KeyGame',
                         'job17583555_task353_2018_05_17_1718_alice_positive_action_cooperative_beta0.25_discount0.8_250k_KeyGame',
                         'job17583555_task354_2018_05_17_1719_alice_positive_action_cooperative_beta0.25_discount0.8_250k_KeyGame',
                         'job17583555_task355_2018_05_17_1718_alice_positive_action_cooperative_beta0.25_discount0.8_250k_KeyGame',
                         'job17583555_task356_2018_05_17_1718_alice_positive_action_cooperative_beta0.25_discount0.8_250k_KeyGame',
                         'job17583555_task210_2018_05_17_1655_alice_unregularized_action_ambivalent_discount0.8_250k_KeyGame',
                         'job17583555_task211_2018_05_17_1653_alice_unregularized_action_ambivalent_discount0.8_250k_KeyGame',
                         'job17583555_task212_2018_05_17_1653_alice_unregularized_action_ambivalent_discount0.8_250k_KeyGame',
                         'job17583555_task213_2018_05_17_1654_alice_unregularized_action_ambivalent_discount0.8_250k_KeyGame',
                         'job17583555_task214_2018_05_17_1654_alice_unregularized_action_ambivalent_discount0.8_250k_KeyGame']
    bob_experiments = ['bob_with_competitive_action_alice',
                       'bob_with_cooperative_state_alice',
                       'bob_with_ambivalent_state_alice']
    
    c = 45
    training_times = {'300k': 300000}
    discounts = [.9]
    for t in training_times.keys():
      for d in discounts:
        i = 0
        for b in range(3):
          for j in range(5):
            replacements = {3: "experiment_name = '{}_discount{}_{}'".format(bob_experiments[b],d,t),
                            5: "alice_experiment = '{}'".format(alice_experiments[i]),
                            32: "training_steps = {} # {}".format(training_times[t],t),
                            37: "                               discount_factor = {},".format(d)}
            gen_config(conf = conf, replacements = replacements, config_ext = str(c))
            i += 1
            c += 1
    
    print('Last config generated: %i' % c)
    
  elif conf == 'alice':
    betas = [.00625, .0125, .025, .05, .1, .2]
    gammas = [.8, .9, .95]
    training_times = {'100k': 100000,
                      '250k': 250000}
    i = 0
    # run competitive experiments first
    for t in training_times.keys():
      for g in gammas:
        for b in betas:
          replacements = {3: "experiment_name = 'alice_negative_action_competitive_beta{}_discount{}_{}'".format(b,g,t),
                          29: "training_steps = {} # {}".format(training_times[t],t),
                          30: "beta = {}".format(-b),
                          31: "gamma = {}".format(g)}
          gen_config(conf = conf, replacements = replacements, config_ext = str(i))
          i += 1
    # then ambivalent
    for t in training_times.keys():
      for g in gammas:
        replacements = {3: "experiment_name = 'alice_unregularized_action_ambivalent_discount{}_{}'".format(g,t),
                        29: "training_steps = {} # {}".format(training_times[t],t),
                        30: "beta = 0",
                        31: "gamma = {}".format(g)}
        gen_config(conf = conf, replacements = replacements, config_ext = str(i))
        i += 1
    # then cooperative
    for t in training_times.keys():
      for g in gammas:
        for b in betas:
          replacements = {3: "experiment_name = 'alice_positive_action_cooperative_beta{}_discount{}_{}'".format(b,g,t),
                          29: "training_steps = {} # {}".format(training_times[t],t),
                          30: "beta = {}".format(b),
                          31: "gamma = {}".format(g)}
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
    