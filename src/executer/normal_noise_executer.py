import numpy as np

from src.executer.base_executer import BaseExecuter


class NormalNoiseExecuter(BaseExecuter):

  # TODO: Change variance of action noise. Currently set to max(action)

  def __init__(self, action_limits):
    super(NormalNoiseExecuter, self).__init__(action_limits=action_limits)

  def _modify(self, action):
    """Normal Noise"""
    action += np.random.normal(np.zeros(self.action_limits.shape[0]), np.max(self.action_limits, axis=1) / 100)
    return action


if __name__ == '__main__':
  exe = NormalNoiseExecuter(action_limits=np.array([[-1, 1], [-2, 2], [-3, 3]]))
  print(exe(
    np.array([0.5, -1, 2])
  ))
