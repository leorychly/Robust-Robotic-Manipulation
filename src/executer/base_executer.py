from abc import ABCMeta, abstractmethod
import numpy as np

class BaseExecuter(metaclass=ABCMeta):
  """This agent implements the Sense-Plan-Act architecture."""

  def __init__(self, action_limits):
    """
    Initialize the action limits of the executer.

    :param action_limits: nd.array
      The min and max values the actions can have in the form of:
      [[a1_min, a1_max], [a2_min, a2_max], [..], ..]
    """
    if action_limits.ndim == 1:
      action_limits = action_limits.reshape(1, -1)
    self.action_limits=action_limits

  def __call__(self, action):
    """
    Modify the agent's action.

    :param action: nd.array
      The agent's action which can be modified.

    :return control action: nd.array
      The action that will actually be executed.
    """
    if not isinstance(action, np.ndarray):
      action = np.array(action)
    action = action.astype(np.float)
    if  hasattr(self, "_modify"):
      action = self._modify(action)
    clipped_action = np.clip(action,
                             a_min=self.action_limits[:, 0],
                             a_max=self.action_limits[:, 1])
    return clipped_action

