import numpy as np

from src.executer.base_executer import BaseExecuter


class ShiftExecuter(BaseExecuter):

  def __init__(self, action_space, shift_scale):
    super(ShiftExecuter, self).__init__(action_space=action_space)
    self.shift_scale = shift_scale

  def _modify(self, action):
    """Normal Noise"""
    action += self.shift_scale
    return action
