import numpy as np
import matplotlib.pyplot as plt

"""
All model were trained without any observation or action manipulation.
"""
### Data
data_obs_noise = {
  "x":     [0, 0.0001, 0.001, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1],
  "y_max": [9343, 9343, 9343, 9348, 9353, 9271, 5902, 1646, 900, 545, 383, 341, 291, 267],
  "y_9k":  [9275, 9275, 9275, 9200, 9121, 8518, 3048, 1270, 916, 588, 446, 368, 319, 309],
  "y_a256_c512": [9118, 9118, 9118, 9197, 9139, 6678, 1914, 1049, 960, 585, 447, 374, 346, 328],
  "y_a512_c256": [9355, 9355, 9355, 9336, 9326, 9222, 3453, 1406, 1062, 579, 399, 304, 286, 264]
}
### Data
data_obs_shift = {
  "x":[-0.1, -0.05, -0.01, -0.005, -0.001, -0.0005, -0.0001, 0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
  "y_max":             [585, 901, 3559, 7960, 9343, 9343, 9343, 9343, 9343, 9343, 9344, 9291, 6013, 940, 639],
  "y_9k":              [729, 921, 7947, 9205, 9275, 9275, 9275, 9275, 9275, 9211, 9204, 8893, 5807, 900, 700],
  "y_a256_c512":       [447, 717, 3020, 7725, 9118, 9118, 9118, 9118, 9118, 9118, 9118, 8966, 5974, 831, 507],
  "y_a512_c256":       [494, 854, 3040, 5285, 9354, 9354, 9355, 9355, 9355, 9355, 9355, 7858, 3655, 959, 646]
}
### Data
data_action_noise = {
  "x": [0, 0.0001, 0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1],
  "y_max":       [9343, 9343, 9343, 9343, 9343, 9343, 9343, 9343, 9343, 9343, 9343, 9343, 9343, 9343, 9344, 9344, 9344, 9347, 9353,  7600, 1877, 827],
  "y_9k":        [9275, 9275, 9275, 9275, 9275, 9275, 9275, 9275, 9275, 9275, 9275, 9275, 9275, 9275, 9275 , 9275, 9276, 9208, 9046, 9025, 1753, 918],
  "y_a256_c512": [9118, 9118, 9118, 9118, 9118, 9118, 9118, 9118, 9118, 9118, 9118, 9118, 9118, 9119, 9117, 9118, 9119, 8822, 8812, 6624, 2270, 927],
  "y_a512_c256": [9355, 9355, 9355, 9355, 9355, 9355, 9355, 9355, 9355, 9355, 9355, 9355, 9355, 9355, 9355, 9355, 9355, 9285, 8722, 7436, 1836, 781]
}
### Data
data_action_shift_old = {
  "x": [-0.1, -0.05, -0.01, -0.005, -0.001, -0.0005, -0.0001, 0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
  "y_max":       [9350, 9345, 9343, 9343, 9343, 9343, 9343, 9343, 9343, 9343, 9343, 9343, 9343, 9345, 8823],
  "y_9k":        [8971, 9126, 9212, 9275, 9275, 9275, 9275, 9275, 9275, 9275, 9275, 9275, 9275, 9276, 9207],
  "y_a256_c512": [9117, 9118, 9118, 9118, 9118, 9118, 9118, 9118, 9118, 9118, 9118, 9118, 9118, 8654, 7118],
  "y_a512_c256": [9355, 9355, 9355, 9355, 9355, 9355, 9355, 9355, 9355, 9355, 9355, 9355, 9354, 9241, 6629]
}
data_action_shift = {
  "x": [-1, -0.8, -0.6, -0.4, -0.2, -0.10, -0.08, -0.06, -0.04, -0.02, -0.01, -0.001, 0, 0.001, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.2, 0.4, 0.6, 0.8, 1.0],
  "y_max":       [303, 777, 1358, 2685, 9281, 9350, 9348, 9346, 9345, 9343, 9343, 9343, 9343, 9343, 9343, 9343, 9344, 9346, 9172, 8823, 6969, 2317, 1167, 539, 263],
  "y_9k":        [323, 738, 1260, 2880, 8424, 8971, 9047, 9065, 9129, 9204, 9212, 9275, 9275, 9275, 9275, 9275, 9275, 9277, 9202, 9207, 9042, 4208, 1090, 549, 262],
  "y_a256_c512": [344, 903, 1349, 2055, 8984, 9117, 9117, 9117, 9118, 9118, 9118, 9118, 9118, 9118, 9118, 9117, 8973, 7487, 6862, 7118, 7102, 3301, 1621, 759, 284],
  "y_a512_c256": [317, 734, 1808, 3702, 9160, 9355, 9355, 9355, 9355, 9355, 9355, 9355, 9355, 9355, 9354, 9354, 9355, 8711, 7784, 6629, 3909, 1734, 875, 536, 274]
}

### Plot
fig1, axes1 = plt.subplots(2, 2)
fig1.suptitle("TD3 Trained with Base Observer and Executer", fontsize=16)

axes1[0, 0].plot(data_obs_noise["x"], data_obs_noise["y_max"], 'red')
axes1[0, 0].plot(data_obs_noise["x"], data_obs_noise["y_9k"], 'orange')
axes1[0, 0].plot(data_obs_noise["x"], data_obs_noise["y_a256_c512"], 'blue')
axes1[0, 0].plot(data_obs_noise["x"], data_obs_noise["y_a512_c256"], 'teal')
axes1[0, 0].set_title("Observation Noise")
axes1[0, 0].set_xlabel("Noise Variance")
axes1[0, 0].set_ylabel("Reward over 1000 Steps (Averaged over 100 Episodes)")
axes1[0, 0].set_xlim([0, 1])
axes1[0, 0].set_ylim([0, 9500])
axes1[0, 0].grid()

axes1[1, 0].plot(data_obs_shift["x"], data_obs_shift["y_max"], 'red')
axes1[1, 0].plot(data_obs_shift["x"], data_obs_shift["y_9k"], 'orange')
axes1[1, 0].plot(data_obs_shift["x"], data_obs_shift["y_a256_c512"], 'blue')
axes1[1, 0].plot(data_obs_shift["x"], data_obs_shift["y_a512_c256"], 'teal')
axes1[1, 0].set_title("Observation Shift")
axes1[1, 0].set_xlabel("Absolute Shift Distance")
axes1[1, 0].set_ylabel("Reward over 1000 Steps (Averaged over 100 Episodes)")
axes1[1, 0].set_xlim([-0.75, 0.75])
axes1[1, 0].set_ylim([0, 9500])
axes1[1, 0].grid()

axes1[0, 1].plot(data_action_noise["x"], data_action_noise["y_max"], 'red')
axes1[0, 1].plot(data_action_noise["x"], data_action_noise["y_9k"], 'orange')
axes1[0, 1].plot(data_action_noise["x"], data_action_noise["y_a256_c512"], 'blue')
axes1[0, 1].plot(data_action_noise["x"], data_action_noise["y_a512_c256"], 'teal')
axes1[0, 1].set_title("Action Noise")
axes1[0, 1].set_xlabel("Noise Variance")
axes1[0, 1].set_ylabel("Reward over 1000 Steps (Averaged over 100 Episodes)")
axes1[0, 1].set_xlim([0, 1])
axes1[0, 1].set_ylim([0, 9500])
axes1[0, 1].grid()

axes1[1, 1].plot(data_action_shift["x"], data_action_shift["y_max"], 'red')
axes1[1, 1].plot(data_action_shift["x"], data_action_shift["y_9k"], 'orange')
axes1[1, 1].plot(data_action_shift["x"], data_action_shift["y_a256_c512"], 'blue')
axes1[1, 1].plot(data_action_shift["x"], data_action_shift["y_a512_c256"], 'teal')
axes1[1, 1].set_title("Action Shift")
axes1[1, 1].set_xlabel("Absolute Shift Distance")
axes1[1, 1].set_ylabel("Reward over 1000 Steps (Averaged over 100 Episodes)")
axes1[1, 1].set_xlim([-0.75, 0.75])
axes1[1, 1].set_ylim([0, 9500])
axes1[1, 1].grid()

plt.show()