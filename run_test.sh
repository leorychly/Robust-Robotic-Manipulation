#!/bin/bash
PYTHONPATH=$PYTHONPATH:$PWD python ./experiments/run_pybulletgym.py --mode test -d ./models/td3_mb -c ./experiments/configs/test_conf.json