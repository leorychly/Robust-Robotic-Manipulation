import numpy as np

from src.executer.base_executer import BaseExecuter

# TODO: Change variance of action noise. Better way than just some factor?


class NoiseExecuter(BaseExecuter):

  def __init__(self, action_space, noise_scale=0.001):
    super(NoiseExecuter, self).__init__(action_space=action_space)
    self.noise_scale = noise_scale

  def _modify(self, action):
    """Normal Noise"""
    action += np.random.normal(np.zeros(self.action_space.shape[0]), self.noise_scale)
    return action