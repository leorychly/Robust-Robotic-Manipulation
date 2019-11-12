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

sys.path.insert(1, "..")
from src.agents.base_agent import RandomAgent
from src.agents.td3.td3_agent import TD3Agent
from src.executer.normal_noise_executer import NormalNoiseExecuter

#TODO: agent funct that returns all self params,
#TODO: save all kwargs + agent args to file

def run_td3_pytorch(args, env):
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
                   observer=None,
                   executer=None,
                   buffer_size=10000)  #  NormalNoiseExecuter(action_limits=action_limits)

  if args.train:
    print("\n========== START TRAINING ==========")
    exp_param = vars(args)
    exp_param["train_steps"] = 1000000
    exp_param["initial_steps"] = 10000
    exp_param["batch_size"] = 64
    exp_param["eval_steps"] = 10

    try:
      agent.load(model_path)
      logging.info(f"The model was loaded from '{model_path}'")
    except Exception as e:
      logging.info(f"No model loaded from '{model_path}'")
    agent.train(env=env,
                seed=args.seed,
                train_steps=exp_param["train_steps"],
                initial_steps=exp_param["initial_steps"],
                model_save_path=model_path,
                results_path=results_path,
                batch_size=exp_param["batch_size"],
                eval_steps=exp_param["eval_steps"],
                eval_freq=1000)
    with open(str(Path(args.directory) / param_path / "experiment_param.json"), 'w') as jsn:
      json.dump({
        "experiment_param": exp_param,
        "agent_param": agent.get_param()
      }, jsn)

    #json.dump(agent.__dict__, str(Path(args.directory) / "params/model_param.json"))


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
  env = gym.make(args.environment)

  if args.agent == "RandomAgent":
    run_random_agent(args, env)

  elif args.agent == "TD3":
    run_td3_pytorch(args, env)



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Run a Robotic manipulation Task.")
  parser.add_argument("-e", "--environment",
                      help="Any environments from PyBullet Gymperium.",
                      default="InvertedDoublePendulumMuJoCoEnv-v0")
  parser.add_argument("-a", "--agent",
                      help="Choose which agent to use for the experiment. ('RandomAgent', 'TD3')",
                      default="TD3")
  parser.add_argument("-exe", "--executor",
                      help="Choose which executor to use for the experiment.",
                      default="BaseExecuter")
  parser.add_argument("-obs", "--observer",
                      help="Choose which observer to use for the experiment.",
                      default="BaseObserver")
  parser.add_argument("-t", "--train",
                      help="If set to 'True' the agent is also trained before evaluation.",
                      default=False)
  parser.add_argument("-l", "--logging",
                      help="Set logging verbosity: 'debug': print all; 'info': print info only",
                      default="info")
  parser.add_argument("-s", "--seed",
                      help="Random Seed",
                      default=int(time.time()), type=np.int)
  parser.add_argument("-d", "--directory",
                      help="The experiment output directory. E.g.: ./experiment_results",
                      default=f"{Path(os.getcwd()) / 'experiment_results/td3'}")

  args = parser.parse_args()

  # Set seeds
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
