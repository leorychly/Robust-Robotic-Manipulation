import numpy as np
from abc import ABCMeta, abstractmethod


class BaseObserver(metaclass=ABCMeta):
  """This agent implements the Sense-Plan-Act architecture."""

  def __init__(self, observation_space):
    """
    Initialize the observation space limits of the observer.

    :param state_space_limits: nd.array
      The min and max values the state can have in the form of:
      [[s1_min, s1_max], [s2_min, s2_max], [..], ..]
    """
    self.obs_space=observation_space

  def __call__(self, state):
    """
    Modify the environment's state.

    :param state: nd.array
      The env's state which will be modified an passed
      to the agent.

    :return observation: nd.array
      The state that the agent will actually observe.
    """
    if not isinstance(state, np.ndarray):
      state = np.array(state)
    state = state.astype(np.float)
    if hasattr(self, "_modify"):
      state = self._modify(state)
    state = np.clip(state, a_min=self.obs_space.low, a_max=self.obs_space.high)
    return state
