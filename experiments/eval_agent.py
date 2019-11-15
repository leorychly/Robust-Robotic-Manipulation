import time
import gym
from absl import logging

from src.utils import read_json, run_agent
from src.agents.td3.td3_agent import TD3Agent

from src.observer.base_observer import BaseObserver
from src.observer.noise_observer import NoiseObserver
from src.observer.shift_observer import ShiftObserver

from src.executer.base_executer import BaseExecuter
from src.executer.noise_executer import NoiseExecuter
from src.executer.shift_executer import ShiftExecuter


def eval_td3_agent(env,
                   model_path,
                   param_path,
                   eval_ep,
                   render_steps):
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
  print("\n========== START EVALUATION ==========\n")
  param = read_json(str(param_path.absolute() / "param_train.json"))
  config = param["experiment_config"]

  observer = config.pop("observer")
  executer = config.pop("executer")

  agent = TD3Agent(env=env,
                   observer=eval(observer)(observation_space=env.observation_space),
                   executer=eval(executer)(action_space=env.action_space),
                   **config)

  agent.load(model_path)
  logging.info(f"The model was loaded from '{model_path}'")

  logging.info(f"Start {eval_ep} evaluation episodes...")
  avg_reward = agent.eval_policy_on_env(eval_gym_env=env, eval_episodes=eval_ep)
  logging.info(f"Average Reward during {eval_ep} evaluation episodes: {avg_reward}.")

  logging.info(f"Start rendering for {eval_ep} episodes...")
  r = run_agent("InvertedDoublePendulumMuJoCoEnv-v0", agent, steps=render_steps)
  logging.info(f"Total Reward during the rendering episode {r}")
