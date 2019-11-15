import os
import json
import time
from pathlib2 import Path
import gym
import pybulletgym
import torch
import numpy as np
import argparse
from absl import logging

from src.utils import read_json
from src.agents.base_agent import RandomAgent

from src.observer.base_observer import BaseObserver
from src.observer.noise_observer import NoiseObserver
from src.observer.shift_observer import ShiftObserver

from src.executer.base_executer import BaseExecuter
from src.executer.noise_executer import NoiseExecuter
from src.executer.shift_executer import ShiftExecuter

from experiments.train_agent import train_td3_agent
from experiments.eval_agent import eval_td3_agent
from experiments.test_agent import test_td3_agent


# TODO: add timestamp + add reading for eval and test to read(param_train_*)


def run_td3(env, args, config):
  """Test and/or evaluate the TD3 algorithm."""
  model_path = Path(args.directory) / "models"
  model_path.mkdir(parents=True, exist_ok=True)
  results_path = Path(args.directory) / "results"
  results_path.mkdir(parents=True, exist_ok=True)
  param_path = Path(args.directory) / "params"
  param_path.mkdir(parents=True, exist_ok=True)

  with open(str(param_path / f"param_{args.mode}.json"), "w") as jsn:
    json.dump({
      "experiment_args": vars(args),
      "experiment_config": config,
    }, jsn)

  if args.mode == "train":
    train_td3_agent(env=env,
                    model_path=model_path,
                    results_path=results_path,
                    **config)

  elif args.mode == "eval":
    eval_td3_agent(env=env,
                   model_path=model_path,
                   param_path=param_path,
                   eval_ep=config["eval_episodes"],  # TODO: how to manage the train_config and the eval_config
                   render_steps=config["render_steps"])

  elif args.mode == "test":
    test_td3_agent(env=env,
                   configurations=config["configurations"],
                   model_path=model_path,
                   param_path=param_path,
                   eval_ep=config["eval_episodes"],
                   seed=config["seed"])

  else:
    print(f"Error: Mode not identified! ({args.mode})")


def run_random_agent(args, env):
  """
  Run a random agent on a given environment.

  :param args: argparse.args
    Arguments defining which observer ad executer to use.
  :param env: gym.Env
  """
  agent = RandomAgent(action_space=env.env.action_space,
                      action_limits=np.array([env.action_space.high, env.action_space.low]).T,
                      observer=eval(args.observer)(),
                      executer=eval(args.executor)())
  env.render()
  obs = env.reset()
  action = agent.run(state=obs)
  obs, rewards, done, _ = env.step(action=action)
  for i in range(1000):
    obs, rewards, done, _ = env.step(env.action_space.sample())
    time.sleep(0.01)


def main(args):
  config = read_json(args.config_file)
  env = gym.make(config["environment"])

  torch.manual_seed(config["seed"] or int(time.time()))
  np.random.seed(config["seed"] or int(time.time()))
  env.seed(config["seed"] or int(time.time()))

  ## Random Agent
  if config["agent"] == "RandomAgent":
    run_random_agent(args, env)

  ## TD3 Agent
  elif config["agent"] == "TD3":
    run_td3(env, args, config)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Run a Robotic manipulation Task.")
  parser.add_argument("-m", "--mode",
                      help="Set 'train', 'eval' or 'test' for training, evaluation"
                           " or testing generalization respectively.",
                      default=False, type=str)
  parser.add_argument("-l", "--logging",
                      help="Set logging verbosity: 'debug': print all; 'info': print info only",
                      default="info", type=str)
  parser.add_argument("-c", "--config_file",
                      help="Training config file. E.g '/Robust-Robotic-Manipulation/experiments/configs/exp01.json'",
                      default=f"{Path(os.getcwd()) / 'experiments/configs/exp01.json'}")
  parser.add_argument("-d", "--directory",
                      help="The model and experiment output directory. E.g.: ./experiment_results",
                      default=f"{Path(os.getcwd()) / 'experiment_results/td3'}",
                      type=str)

  args = parser.parse_args()

  if args.logging is "debug":
    logging.set_verbosity(logging.DEBUG)
  elif args.logging is "info":
    logging.set_verbosity(logging.INFO)
  logging._warn_preinit_stderr = 0

  t0 = time.time()
  main(args)
  print(f"Execution took {time.time() - t0:.2f} seconds.")
