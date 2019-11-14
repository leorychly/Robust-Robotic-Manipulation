import numpy as np

from src.observer.base_observer import BaseObserver


class NoiseObserver(BaseObserver):

  def __init__(self, observation_space, noise_scale):
    super(NoiseObserver, self).__init__(observation_space=observation_space)
    self.noise_scale = noise_scale

  def _modify(self, state):
    state += np.random.normal(np.zeros(state.shape), self.noise_scale)
    return state
