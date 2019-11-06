import gym
import pybulletgym
import time
import argparse
from absl import logging

from src.observer.base_observer import BaseObserver
from src.executer.base_executer import BaseExecuter
from src.agents.base_agent import BaseAgent


def main(args):
  env = gym.make(args.environment)
  env.render()

  agent = BaseAgent(actions_space=env.env.action_space,
                    observer=eval(args.observer)(),
                    executer=eval(args.executor)())

  obs = env.reset()
  obs, rewards, done, _ = env.step(agent.plan(obs))

  for i in range(1000):
    obs, rewards, done, _ = env.step(env.action_space.sample())
    time.sleep(0.01)


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
  parser.add_argument("-l", "--logging",
                      help="Set logging verbosity: 'debug': print all; 'info': print info only",
                      default="info")
  args = parser.parse_args()

  if args.logging is "debug":
    logging.set_verbosity(logging.DEBUG)
  elif args.logging is "info":
    logging.set_verbosity(logging.INFO)
  logging._warn_preinit_stderr = 0

  t0 = time.time()
  main(args)
  print(f"Execution took {time.time() - t0:.2f} seconds.")
