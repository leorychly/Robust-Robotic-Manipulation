#!/bin/bash
virtualenv -p python3.6 ./venv
source ./python/venv/bin/activate && pip install -r utils/requirements.txt