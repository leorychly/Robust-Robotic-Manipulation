import os
import sys
from pathlib2 import Path
import gym
import time
import torch
import numpy as np
import argparse
from absl import logging

sys.path.insert(1, "../")
from src.agents.base_agent import RandomAgent
from src.agents.td3.td3_agent_pyt import TD3Agent


def run_td3_pytorch(args, env):
  base_dir = Path(os.getcwd()).parent
  model_path = base_dir / "experiment_results/td3/models"
  model_path.mkdir(parents=True, exist_ok=True)
  results_path = base_dir / "experiment_results/td3/results"
  results_path.mkdir(parents=True, exist_ok=True)

  agent = TD3Agent(env=env)

  if args.train:
    try:
      agent.load(model_path)
      logging.info(f"The model was loaded from '{model_path}'")
    except Exception as e:
      logging.info(f"No model loaded from '{model_path}'"
                   f"\n\t=> Error while loading: {e}")
    agent.train(env=env,
                seed=args.seed,
                train_steps=1e5,
                initial_steps=1e4,
                model_save_path=model_path,
                results_path=results_path,
                batch_size=100,
                eval_steps=100,
                eval_freq=1000)

  else:
    agent.load(model_path)
    logging.info(f"The model was loaded from '{model_path}'")
    env.render()
    obs = env.reset()
    action = agent.run(state=obs)
    obs, rewards, done, _ = env.step(action=action)
    for i in range(1000):
      obs, rewards, done, _ = env.step(env.action_space.sample())
      time.sleep(0.01)


def run_random_agent(args, env):
  agent = RandomAgent(action_space=env.env.action_space,
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
  env = gym.make(args.environment)

  #run(args, env)
  run_td3_pytorch(args, env)
  #test_td3_pytorch(args, env)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Run a Robotic manipulation Task.")
  parser.add_argument("-e", "--environment",
                      help="Any environments from PyBullet Gymperium.",
                      default="InvertedDoublePendulumMuJoCoEnv-v0")
  parser.add_argument("-a", "--agent",
                      help="Choose which agent to use for the experiment.",
                      default="DQNAgent")
  parser.add_argument("-exe", "--executor",
                      help="Choose which executor to use for the experiment.",
                      default="BaseExecuter")
  parser.add_argument("-obs", "--observer",
                      help="Choose which observer to use for the experiment.",
                      default="BaseObserver")
  parser.add_argument("-t", "--train",
                      help="If set to 'True' the agent is also trained before evaluation.",
                      default=True)
  parser.add_argument("-l", "--logging",
                      help="Set logging verbosity: 'debug': print all; 'info': print info only",
                      default="info")
  parser.add_argument("-s", "--seed",
                      help="Random Seed",
                      default="42", type=np.int)
  args = parser.parse_args()

  # Set seeds
  #tf.random.set_seed(args.seed)
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  if args.logging is "debug":
    logging.set_verbosity(logging.DEBUG)
  elif args.logging is "info":
    logging.set_verbosity(logging.INFO)
  logging._warn_preinit_stderr = 0

  t0 = time.time()
  main(args)
  print(f"Execution took {time.time() - t0:.2f} seconds.")
