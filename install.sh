#!/bin/bash
virtualenv -p python3.6 ./venv
source ./venv/bin/activate && pip install -r utils/requirements.txt