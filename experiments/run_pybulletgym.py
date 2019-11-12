import os
import sys
import json
from pathlib2 import Path
import gym
import pybulletgym
import time
import torch
import numpy as np
import argparse
from absl import logging

#sys.path.insert(1, "..")
from src.utils import read_config
from src.agents.base_agent import RandomAgent
from src.agents.td3.td3_agent import TD3Agent
from src.observer.base_observer import BaseObserver
from src.executer.base_executer import BaseExecuter
from src.executer.normal_noise_executer import NormalNoiseExecuter


def run_td3(env, args, config):
  """Test and/or evaluate the TD3 RL algorithm."""
  model_path = Path(args.directory) / "models"
  model_path.mkdir(parents=True, exist_ok=True)
  results_path = Path(args.directory) / "results"
  results_path.mkdir(parents=True, exist_ok=True)
  param_path = Path(args.directory) / "params"
  param_path.mkdir(parents=True, exist_ok=True)

  action_limits = np.array([env.action_space.low, env.action_space.high]).T
  agent = TD3Agent(env=env,
                   action_limits=action_limits,
                   actor_layer=config["actor_layer"],
                   critic_layer=config["critic_layer"],
                   actor_lr=config["actor_lr"],
                   critic_lr=config["critic_lr"],
                   observer=eval(config["observer"])(),
                   executer=eval(config["executer"])(action_limits=action_limits),
                   buffer_size=config["buffer_size"])  #  NormalNoiseExecuter(action_limits=action_limits)

  if args.train:
    print("\n========== START TRAINING ==========")
    exp_param = vars(args)
    exp_param["train_steps"] = config["train_steps"]
    exp_param["initial_steps"] = config["initial_steps"]
    exp_param["batch_size"] = config["batch_size"]
    exp_param["eval_steps"] = config["eval_steps"]
    with open(str(param_path / "experiment_param.json"), 'w') as jsn:
      json.dump({
        "experiment_param": exp_param,
        "agent_param": agent.get_param()
      }, jsn)

    try:
      agent.load(model_path)
      logging.info(f"The model was loaded from '{model_path}'")
    except Exception as e:
      logging.info(f"No model loaded from '{model_path}'")
    agent.train(env=env,
                seed=config["seed"],
                train_steps=exp_param["train_steps"],
                initial_steps=exp_param["initial_steps"],
                model_save_path=model_path,
                results_path=results_path,
                batch_size=exp_param["batch_size"],
                eval_steps=exp_param["eval_steps"],
                eval_freq=1000)

  else:
    print("\n========== START EVALUATION ==========")
    agent.load(model_path)
    logging.info(f"The model was loaded from '{model_path}'")
    # Run Evaluation
    eval_ep = 10
    logging.info(f"Starting {eval_ep} evaluation episodes...")
    avg_reward = agent.eval_policy_on_env(eval_gym_env=env, eval_episodes=eval_ep)
    logging.info(f"Average Reward during {eval_ep} evaluation episodes: {avg_reward}.")
    # Render
    run_agent(args.environment, agent, steps=1000)


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
  config = read_config(args.config_file)
  # Set seeds
  torch.manual_seed(config["seed"])
  np.random.seed(config["seed"])

  # Create Environment
  env = gym.make(config["environment"])

  # Random Agent
  if config["agent"] == "RandomAgent":
    run_random_agent(args, env)

  # TD3 Agent
  elif config["agent"] == "TD3":
    run_td3(env, args, config)



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Run a Robotic manipulation Task.")
  parser.add_argument("-t", "--train",
                      help="If set to 'True' the agent is also trained before evaluation.",
                      default=False, type=str)
  parser.add_argument("-l", "--logging",
                      help="Set logging verbosity: 'debug': print all; 'info': print info only",
                      default="info", type=str)
  parser.add_argument("-c", "--config_file",
                      help="Path or file to experiment config file(s). E.g '/Robust-Robotic-Manipulation/experiments/configs'",
                      default=f"{Path(os.getcwd()) / 'experiments/configs/exp01.json'}")
  parser.add_argument("-d", "--directory",
                      help="The experiment output directory. E.g.: ./experiment_results",
                      default=f"{Path(os.getcwd()) / 'experiment_results/td3_10'}",
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
