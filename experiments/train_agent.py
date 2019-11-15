import json
from absl import logging

from src.agents.td3.td3_agent import TD3Agent

from src.observer.base_observer import BaseObserver
from src.observer.noise_observer import NoiseObserver
from src.observer.shift_observer import ShiftObserver

from src.executer.base_executer import BaseExecuter
from src.executer.noise_executer import NoiseExecuter
from src.executer.shift_executer import ShiftExecuter


def train_td3_agent(env,
                    model_path,
                    results_path,
                    observer,
                    executer,
                    **kwargs):
  print("\n========== START TRAINING ==========\n")

  agent = TD3Agent(env=env,
                   actor_layer=kwargs["actor_layer"],
                   critic_layer=kwargs["critic_layer"],
                   actor_lr=kwargs["actor_lr"],
                   critic_lr=kwargs["critic_lr"],
                   observer=eval(observer)(observation_space=env.observation_space),
                   executer=eval(executer)(action_space=env.action_space),
                   buffer_size=kwargs["buffer_size"],
                   discount=kwargs["discount"],
                   tau=kwargs["tau"],
                   policy_noise=kwargs["policy_noise"],
                   noise_clip=kwargs["noise_clip"],
                   policy_freq=kwargs["policy_freq"])

  try:
    agent.load(model_path)
    logging.info(f"The model was loaded from '{model_path}'")
  except Exception as e:
    logging.info(f"No model loaded from '{model_path}'")

  agent.train(env=env,
              train_steps=kwargs["train_steps"],
              initial_steps=kwargs["initial_steps"],
              model_save_path=model_path,
              results_path=results_path,
              expl_noise=kwargs["expl_noise"],
              batch_size=kwargs["batch_size"],
              eval_steps=kwargs["eval_steps"])
