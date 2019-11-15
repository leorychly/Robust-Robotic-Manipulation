#!/bin/bash
PYTHONPATH=$PYTHONPATH:$PWD python ./experiments/run_pybulletgym.py --mode test -d ./models/td3_max -c ./experiments/configs/test_conf.json
PYTHONPATH=$PYTHONPATH:$PWD python ./experiments/run_pybulletgym.py --mode test -d ./models/td3_9k -c ./experiments/configs/test_conf.json
PYTHONPATH=$PYTHONPATH:$PWD python ./experiments/run_pybulletgym.py --mode test -d ./models/td3_a256_c512 -c ./experiments/configs/test_conf.json
PYTHONPATH=$PYTHONPATH:$PWD python ./experiments/run_pybulletgym.py --mode test -d ./models/td3_a512_c256 -c ./experiments/configs/test_conf.json
