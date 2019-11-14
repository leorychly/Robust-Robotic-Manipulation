import numpy as np

from src.observer.base_observer import BaseObserver


class ShiftObserver(BaseObserver):

  def __init__(self, observation_space, shift_scale):
    super(ShiftObserver, self).__init__(observation_space=observation_space)
    self.shift_scale = shift_scale

  def _modify(self, state):
    state += self.shift_scale
    return state
