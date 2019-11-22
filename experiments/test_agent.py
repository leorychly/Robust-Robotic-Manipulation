import time
import pprint
from absl import logging
import numpy as np
import torch
import gym

from src.utils import read_json, run_agent
from src.agents.td3.td3_agent import TD3Agent
from src.agents.td3_mb.td3_agent_mb import TD3AgentMB

from src.observer.base_observer import BaseObserver
from src.observer.noise_observer import NoiseObserver
from src.observer.shift_observer import ShiftObserver

from src.executer.base_executer import BaseExecuter
from src.executer.noise_executer import NoiseExecuter
from src.executer.shift_executer import ShiftExecuter


def test_td3_agent(env,
                   model_path,
                   param_path,
                   eval_ep,
                   configurations,
                   use_model=False,
                   seed=False,
                   render_last=False):
  """
  Test the TD3 Agent on randomly initialized environments and render the result.

  :param env: gym.Env
    OpenAI Gym environment to test in.
  :param model_path: String
    Path to the saved TD3 model.
  :param param_path: String
    Path to the parameters of the saved TD3 model.
  :param eval_ep: Int
    Number of episoed to evaluate the agent.
  :param render_steps: Int
    Number of steps to render the agent acting in the environment.
  """
  print("\n========== START TESTING GENERALIZATION ==========\n")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"\Running computation on: '{device}'\n")
  logging.info(f"The model loaded from '{model_path}'")

  train_param = read_json(str(param_path.absolute() / "param_train.json"))
  train_config = train_param["experiment_config"]
  del train_config["observer"]
  del train_config["executer"]

  results = []
  for obs, exe in configurations:
    agent = TD3AgentMB(env=env,
                     observer=eval(obs),
                     executer=eval(exe),
                     device=device,
                     use_model=use_model,
                     **train_config)
    agent.load(model_path)
    avg_reward = agent.eval_policy_on_env(eval_gym_env=env, eval_episodes=eval_ep, seed=seed)
    results.append((obs, exe, avg_reward))
    logging.info(f"Average Reward during {eval_ep} evaluation episodes: {avg_reward}.")

  #pp = pprint.PrettyPrinter(indent=4)
  #pp.pprint(f"\nResults:\n{results}")
