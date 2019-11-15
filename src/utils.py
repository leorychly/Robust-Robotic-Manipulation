import os
import time
import gym
import pybulletgym
from pathlib2 import Path
from os.path import isfile
import json

def read_json(json_dir):
  """
  Read JSON file from directory.

  :param json_dir: String
    File path.
  :return: Dict
    Data from JSON file as python dict.
  """
  if isfile(json_dir):
    with open(json_dir) as jfile:
        data = json.load(jfile)
        return data
  else:
    print("Error: This is not a JSON file!")


def run_agent(env_name, agent, steps):
  """
  Run an agent on an environment for _ steps.

  :param env_name: String
    Name of the environment.
  :param agent: BaseAgent
    A trained agent.
  :param steps: Int
    Number of steps to run the agent.
  """
  env = gym.make(env_name)
  env.render()
  obs = env.reset()
  action = agent.run(state=obs)
  obs, rewards, done, _ = env.step(action=action)
  for i in range(steps):
    action = agent.run(state=obs)
    obs, rewards, done, _ = env.step(action)
    time.sleep(0.01)


if __name__ == '__main__':
    p = Path(os.getcwd()).parent / "experiments/configs/exp01.json"
    data = read_json(p)
    print(data)