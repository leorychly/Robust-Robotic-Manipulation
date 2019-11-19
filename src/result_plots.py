import numpy as np
import matplotlib.pyplot as plt


def observation_noise():
  x = [0, 0.0001, 0.001, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1]

  y_max =       [9343, 9343, 9343, 9348, 9353, 9271, 5902, 1646, 900, 545, 383, 341, 291, 267]
  y_9k =        [9275, 9275, 9275, 9200, 9121, 8518, 3048, 1270, 916, 588, 446, 368, 319, 309]
  y_a256_c512 = [9118, 9118, 9118, 9197, 9139, 6678, 1914, 1049, 960, 585, 447, 374, 346, 328]
  y_a512_c256 = [9355, 9355, 9355, 9336, 9326, 9222, 3453, 1406, 1062, 579, 399, 304, 286, 264]

  plt.plot(x, y_max, 'red')
  plt.plot(x, y_9k, 'orange')
  plt.plot(x, y_a256_c512, "magenta")
  plt.plot(x, y_a512_c256, "teal")

  plt.title("TD3 Robustness vs Observation Noise\n(2 Layer Actor Critic)")
  plt.xlabel("Noise Variance")
  plt.ylabel("Total Reward over 1000 Steps")
  plt.xlim([0, 0.2])
  plt.grid()
  plt.show()


def observation_shift():
  x = [-0.1, -0.05, -0.01, -0.005, -0.001, -0.0005, -0.0001, 0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]

  y_max =       [585, 901, 3559, 7960, 9343, 9343, 9343, 9343, 9343, 9343, 9344, 9291, 6013, 940, 639]
  y_9k =        [729, 921, 7947, 9205, 9275, 9275, 9275, 9275, 9275, 9211, 9204, 8893, 5807, 900, 700]
  y_a256_c512 = [447, 717, 3020, 7725, 9118, 9118, 9118, 9118, 9118, 9118, 9118, 8966, 5974, 831, 507]
  y_a512_c256 = [494, 854, 3040, 5285, 9354, 9354, 9355, 9355, 9355, 9355, 9355, 7858, 3655, 959, 646]

  plt.plot(x, y_max, 'red')
  plt.plot(x, y_9k, 'orange')
  plt.plot(x, y_a256_c512, "magenta")
  plt.plot(x, y_a512_c256, "teal")

  plt.title("TD3 Robustness vs Observation Shift\n(2 Layer Actor Critic)")
  plt.xlabel("Absolute Shift Distance")
  plt.ylabel("Total Reward over 1000 Steps")
  #plt.xticks(x)
  #plt.xscale("log")
  #plt.xlim([0, 0.2])
  plt.grid()
  plt.show()


def action_noise():
  x = [0, 0.0001, 0.0002, 0.0004, 0.0006, 0.0008,
       0.001, 0.002, 0.004, 0.006, 0.008,
       0.01, 0.02, 0.04, 0.06, 0.08,
       0.1, 0.2, 0.4, 0.6, 0.8, 1]

  y_max = [9343, 9343, 9343, 9343, 9343, 9343, 9343, 9343, 9343, 9343, 9343, 9343, 9343, 9343,
           9344, 9344, 9344, 9347, 9353,  7600, 1877, 827]
  y_9k = [9275, 9275, 9275, 9275, 9275, 9275, 9275, 9275, 9275, 9275, 9275, 9275, 9275, 9275, 9275 , 9275,
          9276, 9208, 9046, 9025, 1753, 918]
  y_a256_c512 = [9118, 9118, 9118, 9118, 9118, 9118, 9118, 9118, 9118, 9118, 9118, 9118, 9118,
                 9119, 9117, 9118, 9119, 8822, 8812, 6624, 2270, 927]
  y_a512_c256 = [9355, 9355, 9355, 9355, 9355, 9355, 9355, 9355, 9355, 9355, 9355, 9355, 9355, 9355, 9355, 9355, 9355,
                 9285, 8722, 7436, 1836, 781]

  plt.plot(x, y_max, 'red')
  plt.plot(x, y_9k, 'orange')
  plt.plot(x, y_a256_c512, "magenta")
  plt.plot(x, y_a512_c256, "teal")

  plt.title("TD3 Robustness vs Action Noise\n(2 Layer Actor Critic)")
  plt.xlabel("Noise Variance")
  plt.ylabel("Total Reward over 1000 Steps")
  # plt.xticks(x)
  # plt.xscale("log")
  # plt.xlim([0, 0.2])
  plt.grid()
  plt.show()



#observation_noise()
#observation_shift()
action_noise()