#!/bin/bash
#sleep 14400 && <command> &

PYTHONPATH=$PYTHONPATH:$PWD python ./experiments/run_pybulletgym.py -t True -d ./experiment_results/td3_01 -c ./experiments/configs/exp01.json

PYTHONPATH=$PYTHONPATH:$PWD python ./experiments/run_pybulletgym.py -t True -d ./experiment_results/td3_02 -c ./experiments/configs/exp02.json

PYTHONPATH=$PYTHONPATH:$PWD python ./experiments/run_pybulletgym.py -t True -d ./experiment_results/td3_03 -c ./experiments/configs/exp03.json
