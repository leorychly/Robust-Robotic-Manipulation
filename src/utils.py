import os
from pathlib2 import Path
from os.path import isfile, isdir, join
import json

def read_config(config_path):
  if isfile(config_path):
    with open(config_path) as jfile:
        data = json.load(jfile)
        return data
  else:
    print("Error: This is not a file!")


if __name__ == '__main__':
    p = Path(os.getcwd()).parent / "experiments/configs/exp01.json"
    data = read_config(p)
    print(data)