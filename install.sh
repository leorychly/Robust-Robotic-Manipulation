#!/bin/bash
virtualenv -p python3.6 ./venv
source ./venv/bin/activate && pip install -r utils/requirements.txt

mkdir ./environments/
cd ./environments/
git clone https://github.com/benelot/pybullet-gym.git
cd pybullet-gym
pip install -e .
