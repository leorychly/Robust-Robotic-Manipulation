#!/bin/bash
PYTHONPATH=$PYTHONPATH:$PWD python ./experiments/run_pybulletgym.py --mode train -d ./experiment_results/action_noise/td3_1 -c ./experiments/configs/train_conf_action_noise.json
PYTHONPATH=$PYTHONPATH:$PWD python ./experiments/run_pybulletgym.py --mode train -d ./experiment_results/action_noise/td3_2 -c ./experiments/configs/train_conf_action_noise.json
PYTHONPATH=$PYTHONPATH:$PWD python ./experiments/run_pybulletgym.py --mode train -d ./experiment_results/action_noise/td3_3 -c ./experiments/configs/train_conf_action_noise.json
