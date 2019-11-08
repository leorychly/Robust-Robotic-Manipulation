import numpy as np

from src.executer.base_executer import BaseExecuter


class NormalNoiseExecuter(BaseExecuter):

  def __init__(self, max_action):
    super(NormalNoiseExecuter, self).__init__(max_action=max_action)

  def __call__(self, action):
    return action

  def _normal_noise(self, action):
    action += np.random.normal(0, np.max(action)).clip(self.max_action)