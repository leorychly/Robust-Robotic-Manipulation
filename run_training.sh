#!/bin/bash

#INVENV=$(python -c 'import sys; print (1 if sys.prefix[-4:] == "venv" else 0)');
#source ./venv/bin/activate;

PYTHONPATH=$PYTHONPATH:$PWD python ./experiments/run_pybulletgym.py --mode train_mb -d ./experiment_results/td3_mb -c ./experiments/configs/train_td3mb.json
