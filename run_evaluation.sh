#!/bin/bash
PYTHONPATH=$PYTHONPATH:$PWD python ./experiments/run_pybulletgym.py --mode eval -d ./models/td3_a512_c256 -c ./experiments/configs/eval_conf.json
